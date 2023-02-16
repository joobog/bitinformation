#!/usr/bin/env python3

import numpy as np

class ConstantFieldException(Exception):
    pass

class Underflow(Exception):
    pass

class IeeeTable:
    def __init__(self):
        self._U = np.uint32
        self._F = np.float64

        i = self._U(0)
        mmin = self._U(0x800000)
        mmax = self._U(0xffffff)
        e = self._F(1)

        self._e = np.zeros(255, dtype=self._F)
        self._v = np.zeros(255, dtype=self._F)

        for i in range(1, 105):
            e *= self._U(2)
            self._e[i + 150] = e
            self._v[i + 150] = e * mmin

        self._e[150] = 1
        self._v[150] = mmin
        e = self._F(1)
        for i in range(1, 150):
            e /= self._U(2)
            self._e[150 - i] = e
            self._v[150 - i] = e * mmin

        self._vmin = self._v[1]
        self._vmax = self._e[254] * mmax

    def search(self, x):
        jl = self._U(0)
        ju = self._U(self._v.size)
        while ju - jl > 1:
            jm = (ju + jl) >> self._U(1)
            if x >= self.v[jm]:
                jl = jm
            else:
                ju = jm
        return jl

    @property
    def vmin(self):
        return self._vmin

    @property
    def vmax(self):
        return self._vmax

    @property
    def v(self):
        return self._v

    @property
    def e(self):
        return self._e


class SimplePacking:
    def __init__(self):
        self._ieee_table = IeeeTable()
        self._U = np.uint64
        self._F = np.float64
        self._last = 127

    def __long_to_ieee(self, x):
        s = x & self._U(0x80000000)
        c = (x & self._U(0x7f800000)) >> self._U(23)
        m = x & self._U(0x007fffff)

        if (c == 0) and (m == 0):
            return 0

        if c == 0:
            m |= self._U(0x800000)
            c = 1
        else:
            m |= self._U(0x800000)

        val = m * self._ieee_table.e[c]
        if s:
            val = -val

        return val


    def __ieee_to_long(self, x):
        s = self._U(0)
        mmax = self._U(0xffffff)
        mmin = self._U(0x800000)
        m = self._U(0)
        e = self._U(0)
        rmmax = self._F(mmax + 0.5)

        if x < 0:
            s = self._U(1)
            x = -x
        if x < self._ieee_table.vmin:
            return s << self._U(31)

        if x > self._ieee_table.vmax:
            raise Exception(f'Number is too large: x {x} > xmax {self._ieee_table.vmax}')

        e = self._ieee_table.search(x)

        x = x / self._ieee_table.e[e]
        while x < mmin:
            x *= self._U(2)
            e -= self._U(1)

        while x > rmmax:
            x /= self._U(2)
            e += self._U(1)

        m = self._U(x + 0.5)
        if m > mmax:
            e += self._U(1)
            m = self._U(0x800000)
        return (s << self._U(31)) | (e << self._U(23)) | (m & self._U(0x7fffff))


    def __ieee_nearest_smaller_to_long(self, x) -> int:
        l = self._U(0)
        e = self._U(0)
        m = self._U(0)
        s = self._U(0)
        mmin = self._U(0x800000)
        y = self._F(0)
        eps = self._F(0)
        if x == 0:
            return self._U(0)
        l = self.__ieee_to_long(x)
        y = self.__long_to_ieee(l)
        if x < y:
            if (x < 0) and (-x < self._ieee_table.vmin):
                l = self._U(0x80800000)
            else:
                e = (l & self._U(0x7f800000)) >> self._U(23)
                m = (l & self._U(0x007fffff)) | self._U(0x800000)
                s = l & self._U(0x80000000)

                if m == mmin:
                    e = e if s else e - 1
                    if e < 1:
                        e = self._U(1)
                    if e > 254:
                        e = self._U(254)
                eps = self._ieee_table.e[e]
                l = self.__ieee_to_long(y - eps)
        else:
            return l
        if x < self.__long_to_ieee(l):
            assert x >= self.__long_to_ieee(l)
        return l


    def __nearest_smaller_ieee_float(self, d):
        l = self.__ieee_nearest_smaller_to_long(d)
        return self.__long_to_ieee(l)


    def __compute_decimal_scale_factor(self, values, bits_per_value):
        vmin = np.min(values)
        vmax = np.max(values)
        if vmin == vmax:
            raise ConstantFieldException()
        unscaled_max = vmax
        unscaled_min = vmin
        f = 2**bits_per_value - 1
        minrange = 2**(-self._last) * f
        maxrange = 2**(self._last) * f
        vrange = vmax - vmin
        decimal = 1.0
        decimal_scale_factor = 0
        # print(f'Range {vrange} minrange {minrange}')
        while vrange < minrange:
            decimal_scale_factor += 1
            decimal *= 10
            vmin = unscaled_min * decimal
            vmax = unscaled_max * decimal
            vrange = (vmax - vmin)
        while vrange > maxrange:
            decimal_scale_factor -= 1
            decimal /= 10.0
            vmin = unscaled_min * decimal
            vmax = unscaled_max * decimal
            vrange = (vmax - vmin)

        # print(f'vmin {vmin} vmax {vmax}')
        return decimal_scale_factor, vmin, vmax


    def __compute_binary_scale_factor(self, vmax, vmin, bits_per_value):
        vrange = vmax - vmin
        dmaxint = self._F(2**bits_per_value - 1)
        if dmaxint >= np.iinfo(self._U).max:
            raise Exception("Out of vrange")
        maxint = self._U(dmaxint)
        if bits_per_value < 1:
            raise Exception("Bits per value < 1")
        zs = 1
        scale = 0
        # print(f'vrange {vrange} zs {zs} dmaxint {dmaxint}')
        # print(f'vmax - vmin = vrange {vmax} {vmin} {vrange}')
        while (vrange * zs) <= dmaxint:
            scale -= 1
            zs *= 2
        # print(scale)
        while (vrange * zs) > dmaxint:
            scale += 1
            zs /= 2
        # print(scale)
        while self._U(vrange * zs + 0.5) <= maxint:
            scale -= 1
            zs *= 2
        # print(scale)
        while self._U(vrange * zs + 0.5) > maxint:
            scale += 1
            zs /= 2
        # print(scale)

        if scale < -self._last:
            # print(scale, -self._last)
            # print(f'vmax {vmax}, vmin {vmin}, bits_per_value {bits_per_value}')
            raise Underflow
        assert scale <= self._last
        return scale


    def encode(self, values, bits_per_value):
        D, scaled_min, scaled_max = self.__compute_decimal_scale_factor(values, bits_per_value)
        nearest = self.__nearest_smaller_ieee_float(scaled_min)
        E = self.__compute_binary_scale_factor(scaled_max, nearest, bits_per_value)
        min_value = np.min(values)
        R = self.__nearest_smaller_ieee_float(min_value)
        R = R*10**D
        # print(f'R {R}, D {D}, E {E}')
        data = ((values * 10**D - R) / 2**E + 0.5).astype(np.uint64)
        return (R, E, D, data)

    def decode(self, R, E, D, values):
        return (values.astype(np.float64) * 2**E + R) / 10**D
