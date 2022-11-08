#!/usr/bin/env python3

import math
import pandas as pd
import numpy as np
import sys
import scipy.stats

import struct
def binary(num):
    return ''.join('{:0>8b}'.format(c) for c in struct.pack('!Q', num))

class BitInformation:
    def __init__(self):
        pass

    def __permute_dim_forward(self, A, dim):
        assert(dim <= A.ndim)
        R = np.moveaxis(A, np.arange(A.ndim), np.roll(np.arange(A.ndim), dim-1))
        return R

    def __binom_confidence(self, n, c):
        '''Returns the probability `p₁` of successes in the binomial distribution (p=1/2) of
        `n` trials with confidence `c`.'''
        p = scipy.stats.norm(loc=0, scale=1.0).interval(c)[1]/(2*math.sqrt(n)) + 0.5
        return min(1.0, p)


    def __binom_free_entropy(self, n, c, base=2):
        '''Returns the free entropy `Hf` associated with `binom_confidence`.'''
        p = self.__binom_confidence(n, c)
        entropy =  1 - scipy.stats.entropy([p, 1-p], base=base)
        return entropy

    def __set_zero_insignificant(self, H, nelements, confidence):
        '''Remove binary information in the vector `H` of entropies that is insignificantly
        different from a random 50/50 by setting it to zero.'''
        Hfree = self.__binom_free_entropy(nelements, confidence)
        for i in range(0, H.size):
            H[i] = 0 if H[i] <= Hfree else H[i]
        return H

    # def __bitpair_count_a_mask(self, A, mask):
    #     T = A.dtype.type
    #     nbits = A.itemsize * 8
    #     C = np.zeros(dtype=T, shape=(nbits, 2, 2))
    #     Auint = A.astype(T)
    #     nelements = Auint.size
    #     for i in range(0, nelements-1):
    #         if not mask[i] or mask[i+1]:
    #             self.__bitpair_count_c_a_b(C, Auint[i], Auint[i+1])
    #     return C

    # def __bitpair_count_c_a_b(self, C, a, b):
    #     T = C.dtype.type
    #     nbits = int(C.itemsize * 8)
    #     mask = T(1)
    #     for i in C.dtype.type(range(0, nbits)):
    #         j = int((a & mask) / (2**i))
    #         k = int((b & mask) / (2**i))
    #         C[int(nbits-i-1), int(j), int(k)] += 1
    #         mask = T(mask * 2)

    # def __bitpair_count_a_b(self, A, B):
    #     T = A.dtype.type
    #     assert(A.size == B.size)
    #     nbits = A.itemsize * 8
    #     C = np.zeros((nbits,2,2), dtype=T)
    #     Auint = A.astype(T)
    #     Buint = B.astype(T)
    #     for (a,b) in zip(Auint, Buint):
    #         self.__bitpair_count_c_a_b(C,a,b)
    #     return C

    def __bitpair_count_a_b_vectorised(self, A, B):
        T = A.dtype.type
        assert(A.size == B.size)
        nbits = A.itemsize * 8
        Auint = np.ndarray(shape=(1, A.size), buffer=A, dtype=T)
        Buint = np.ndarray(shape=(1, B.size), buffer=B, dtype=T)

        shifts = np.ndarray(shape=(nbits, 1), buffer=np.arange(start=0, stop=nbits, dtype=T), dtype=T)
        mask = T(0x1) << shifts

        j = (Auint & mask) >> shifts
        k = (Buint & mask) >> shifts

        jr = np.reshape(j, newshape=(j.size))
        kr = np.reshape(k, newshape=(k.size))
        c = np.repeat(np.arange(start=0, stop=nbits)[::-1], Auint.size)

        df_tmp = pd.DataFrame({'c':c, 'j':jr, 'k':kr})
        df = df_tmp.groupby(['c', 'j', 'k']).aggregate(count=('c', 'count'))
        C = np.zeros((nbits,2,2), dtype=T)
        C[df.index.get_level_values('c'), df.index.get_level_values('j'), df.index.get_level_values('k')] = df['count']
        return C

    def __bitpair_count_a_b_partially_vectorised(self, A, B):
        T = A.dtype.type
        assert(A.size == B.size)
        nbits = T(A.itemsize * 8)
        Auint = np.ndarray(shape=(A.size), buffer=A, dtype=T)
        Buint = np.ndarray(shape=(B.size), buffer=B, dtype=T)
        shifts = np.arange(start=0, stop=nbits, dtype=T)

        df_parts = list()
        for shift in shifts:
            mask = T(0x1) << shift
            j = (Auint & mask) >> shift
            k = (Buint & mask) >> shift
            c = np.repeat(nbits-shift-T(1), Auint.size)
            tmp = pd.DataFrame({'c':c, 'j':j, 'k':k}).groupby(['c', 'j', 'k']).aggregate(count=('c', 'count'))
            df_parts.append(tmp)

        df = pd.concat(df_parts).groupby(['c', 'j', 'k']).aggregate(count=('count', 'sum'))
        C = np.zeros((nbits,2,2), dtype=T)

        C[df.index.get_level_values('c'), df.index.get_level_values('j'), df.index.get_level_values('k')] = df['count']
        return C

    def __mutual_information2(self, p, base=2):
        nx = p[0].size
        ny = p[0].size
        py = p.sum(axis=0)
        px = p.sum(axis=1)
        M = p.dtype.type(0)
        for j in range(0, ny):
            for i in range(0, nx):
                if p[i,j] > 0:
                    M += p[i,j] * np.log(p[i,j] / px[i] / py[j])
        M /= np.log(base)
        return M

    def __mutual_information(self, A, B, szi=True, confidence=0.99):
        '''Compute information content for each bit'''
        nelements = A.size
        nbits = A.itemsize * 8
        # C = self.__bitpair_count_a_b(A, B)                    # very slow
        # C = self.__bitpair_count_a_b_vectorised(A, B)         # fast, high memory usage
        C = self.__bitpair_count_a_b_partially_vectorised(A, B) # fast, moderate memory usage
        M = np.zeros(nbits, dtype=np.float64)
        P = np.zeros((2,2))
        for i in range(0, nbits):
            for j in [0, 1]:
                for k in [0, 1]:
                    P[j,k] = C[i,j,k] / nelements
            M[i] = self.__mutual_information2(P)

        # remove information that is insignificantly different from a random 50/50 experiment
        if szi:
            self.__set_zero_insignificant(M, nelements, confidence)
        return M

    def bitinformation(self, A, set_zero_insignificant=True, confidence=0.99):
        if type(A) is not np.ndarray:
            raise Exception(f'Expect numpy.ndarray as parameter but got {type(A)}')

        uintxx = 'uint' + str(A.itemsize*8)
        A_uint = np.frombuffer(A, uintxx)
        A1view = A_uint[:-1]
        A2view = A_uint[1:]
        M = self.__mutual_information(A1view, A2view, confidence=confidence)
        return M

    def compare_data(self, data1, data2, set_zero_insignificant=True, confidence=0.99):
        ''' Return 0 if data are equal or 1 if they are not. '''
        biinfo = BitInformation()
        bi1 = biinfo.bitinformation(data1, set_zero_insignificant=set_zero_insignificant, confidence=confidence)
        bi2 = biinfo.bitinformation(data2, set_zero_insignificant=set_zero_insignificant, confidence=confidence)

        uintxx = 'uint' + str(data1.itemsize*8)
        data1_uint = np.frombuffer(data1, uintxx)
        data2_uint = np.frombuffer(data2, uintxx)

        # create a mask for removing the least significant zeros,
        # e.g., 1111.1111.1000.0000
        T = data1_uint.dtype.type
        mask = T(0x0)
        for a, b in zip(reversed(bi1), reversed(bi2)):
            if a == 0 and b == 0:
                mask = T(mask * 2)
                mask = mask | T(0x1)
            else:
                break
        mask = ~mask

        for a, b in zip(data1_uint, data2_uint):
            a_masked = a & mask
            b_masked = b & mask
            if a_masked != b_masked:
                return 0
        return 1

    def compare_grib_files(fn1, fn2, set_zero_insignificant=True, confidence=0.99):
        ''' Return 0 if files are equal, otherwise return index of not
        equal messages. '''
        f1 = open(fn1, 'r')
        f2 = open(fn2, 'r')
        count = 1
        while True:
            h1 = codes_grib_new_from_file(f1)
            h2 = codes_grib_new_from_file(f2)
            if not (h1 and h2):
                break
            vals1 = codes_get_values(h1)
            vals2 = codes_get_values(h2)
            codes_release(h1)
            codes_release(h2)
            if compare_data(vals1, vals2, set_zero_insignificant=set_zero_insignificant, confidence=confidence) != 0:
                return count
            count += 1
        f1.close()
        f2.close()
        return 0

