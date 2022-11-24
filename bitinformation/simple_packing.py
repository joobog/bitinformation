#!/usr/bin/env python3

import numpy as np

class ConstantFieldException(Exception):
    pass

class IeeeTable:
    def __init__(self):
        self._U = np.uint64
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

        self._vmin   = self._v[1]
        self._vmax   = self._e[254] * mmax

    def search(self, x):
        U = np.uint64
        F = np.float64
        jl = self._U(0)
        ju = self._U(self._v.size)
        while (ju - jl > 1):
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
        print(type(x))
        s = x & self._U(0x80000000)
        c = (x & self._U(0x7f800000)) >> self._U(23)
        m = x & self._U(0x007fffff)

        if (c == 0) and (m == 0):
            return 0

        if (c == 0):
            m |= self._U(0x800000)
            c = 1
        else:
            m |= self._U(0x800000)

        val = m * self._ieee_table.e[c]
        if s:
            val = -val

        return val


    def __ieee_to_long(self, x):
        s    = self._U(0)
        mmax = self._U(0xffffff)
        mmin = self._U(0x800000)
        m    = self._U(0)
        e    = self._U(0)
        rmmax = self._F(mmax + 0.5)

        if x < 0:
            s = self._U(1)
            x = -x
        if x < self._ieee_table.vmin:
            return (s << self._U(31))

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
        if (x < y):
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
        if x < long_to_ieee(l):
            assert x >= long_to_ieee(l)
        return l


    def __nearest_smaller_ieee_float(self, d):
        l = self.__ieee_nearest_smaller_to_long(d)
        print(type(l))
        return self.__long_to_ieee(l)


    def __compute_decimal_scale_factor(self, values, bits_per_value):
        min = np.min(values)
        max = np.max(values)
        unscaled_max = max
        unscaled_min = min
        f = 2**bits_per_value
        minrange = 2**(-self._last) * f
        maxrange = 2**(self._last) * f
        range = max - min
        decimal = 1.0
        decimal_scale_factor = 0.0
        while range < minrange:
            decimal_scale_factor += 1
            decimal *= 10
            min = unscaled_min * decimal
            max = unscaled_max * decimal
            range = (max - min)
        while range > maxrange:
            decimal_scale_factor -= 1
            decimal /= 10.0
            min = unscaled_min * decimal
            max = unscaled_max * decimal
            range = (max - min)
        return decimal_scale_factor


    def __compute_binary_scale_factor(self, max, min, bits_per_value):
        range = max - min;
        dmaxint = self._F(2**bits_per_value - 1)
        if dmaxint >= np.iinfo(self._U).max:
            raise Exception("Out of range")
        maxint = self._U(dmaxint)
        if bits_per_value < 1:
            raise Exception("Bits per value < 1")
        if range == 0:
            raise ConstantFieldException()
        zs    = 1
        scale = 0
        while (range * zs) <= dmaxint:
            scale -= 1
            zs *= 2
        while (range * zs) > dmaxint:
            scale += 1;
            zs /= 2;
        while self._U(range * zs + 0.5) <= maxint:
            scale -= 1
            zs *= 2
        while self._U(range * zs + 0.5) > maxint:
            scale += 1
            zs /= 2

        if scale < -self._last:
            raise Exception("Underflow")
        assert scale <= self._last
        return scale


    def encode(self, values, bits_per_value):
        E = self.__compute_binary_scale_factor(np.max(values), np.min(values), bits_per_value)
        D = self.__compute_decimal_scale_factor(values, bits_per_value)
        min_value = np.min(values)
        R = self.__nearest_smaller_ieee_float(min_value)
        R = R*10**D
        # print(f'R {R}, D {D}, E {E}')
        data = ((values * 10**D - R) / 2**E + 0.5).astype(np.uint64)
        return data


if '__main__' == __name__:
    bits_per_value = 16
    values = np.array([0.12, 0.23, 0.42, 0.54, 0.12, 0.30, 0.12], dtype = np.float64 )
    sp = SimplePacking()
    data = sp.encode(values, bits_per_value)
    print(data)
