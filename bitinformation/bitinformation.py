#!/usr/bin/env python3

import math
import pandas as pd
import numpy as np
import sys
import scipy.stats
from timeit import default_timer as timer


# def permute_dim_forward(self, A, dim):
    # assert(dim <= A.ndim)
    # R = np.moveaxis(A, np.arange(A.ndim), np.roll(np.arange(A.ndim), dim-1))
    # return R

def binom_confidence(n, c):
    '''Returns the probability `p₁` of successes in the binomial distribution (p=1/2) of
    `n` trials with confidence `c`.'''
    p = scipy.stats.norm(loc=0, scale=1.0).interval(c)[1]/(2*math.sqrt(n)) + 0.5
    return min(1.0, p)

def binom_free_entropy(n, c, base=2):
    '''Returns the free entropy `Hf` associated with `binom_confidence`.'''
    p = binom_confidence(n, c)
    entropy =  1 - scipy.stats.entropy([p, 1-p], base=base)
    return entropy

class Analyser:
    @property
    def data(self):
        pass

    @data.setter
    def data(self, data):
        pass

    @property
    def mask(self):
        pass

    @property
    def nbits_used(self):
        pass

    def bitinformation(self):
        pass

    def cleaned_data(self):
        # inf = self.bitinformation()
        data = self.data
        OT = data.dtype.type
        uintxx = 'uint' + str(data.itemsize*8)
        data_uint = np.frombuffer(data, uintxx)
        mask = self.mask
        if self._verbose > 0:
            print('Used bit mask: ', bin(mask))
        a = data_uint & mask
        return np.frombuffer(a, OT)

    def compare(self):
        pass


class ManualAnalyser(Analyser):
    def __init__(self, data, mask):
        self._data = data
        self._mask = mask
        self._verbose = 0

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data

    @property
    def mask(self):
        return self._mask

    @property
    def nbits_used(self):
        return np.sum(self.bitinformation()).astype(np.int64)

    def bitinformation(self):
        U = self._data.dtype.type
        mask = self._mask
        nbits = self._data.itemsize * 8
        bi = np.zeros(nbits)
        for i in np.flip(np.arange(start=0, stop=nbits)):
            bi[i] = mask & U(0x1)
            mask = mask >> U(0x1)
        return bi

    def compare(self, data):
        if self._verbose > 0:
            df = pd.DataFrame({'Value1':data, 'Value2':cleaned_data, 'diff':(data - cleaned_data)}).reset_index()
            df['index'] += 1
            df = df[df['diff'] != 0.0]
            if not df.empty:
                print(df)
        return np.all(self._data == data)



class BitInformationAnalyser(Analyser):
    def __init__(self, data, verbose=0, szi=True, confidence=0.99, precision=0.99):
        self._data = data
        self._verbose = verbose
        self._szi=szi
        self._confidence=confidence
        self._parameter_changed = True
        self._precision = precision


    def __szi(self, H, nelements, confidence):
        '''Remove binary information in the vector `H` of entropies that is insignificantly
        different from a random 50/50 by setting it to zero.'''
        Hfree = binom_free_entropy(nelements, confidence)
        for i in range(0, H.size):
            H[i] = 0 if H[i] <= Hfree else H[i]
        return H

    def __bitpair_count_a_b_fast(self, Auint):
        T = Auint.dtype.type
        nbits = T(Auint.itemsize * 8)
        shifts = np.arange(start=0, stop=nbits, dtype=T)
        C = np.zeros((nbits,2,2), dtype=np.uint64)
        for shift in shifts:
            v = (Auint >> shift).astype(dtype=np.ubyte) & np.ubyte(0x1)
            j = v[:-1]
            k = v[1:]
            idx = j << np.ubyte(1) | k
            unique, counts = np.unique(idx, return_counts=True)
            idx_c = nbits-shift-T(1)
            for u, c in zip(unique, counts):
                idx_j = u >> np.ubyte(1) & np.ubyte(0x1)
                idx_k = u & np.ubyte(0x1)
                if self._verbose > 3:
                    print(f'{{pos:{idx_c}, seq:{idx_j}{idx_k}, count:{c}}}')
                C[idx_c, idx_j, idx_k] = c
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

    def __mutual_information(self, A, szi=True, confidence=0.99):
        '''Compute information content for each bit'''
        nelements = A.size
        nbits = A.itemsize * 8
        start = timer()
        C = self.__bitpair_count_a_b_fast(A) # fast, moderate memory usage
        stop = timer()
        if self._verbose > 3:
            print(f'Calculation runtime: {stop - start} seconds')
        M = np.zeros(nbits, dtype=np.float64)
        P = np.zeros((2,2))
        for i in range(0, nbits):
            for j in [0, 1]:
                for k in [0, 1]:
                    P[j,k] = C[i,j,k] / nelements
            M[i] = self.__mutual_information2(P)

        # remove information that is insignificantly different from a random 50/50 experiment
        if szi:
            self.__szi(M, nelements, confidence)
        return M


    def __compute_bitinformation(self):
        if type(self._data) is not np.ndarray:
            raise Exception(f'Expect numpy.ndarray as parameter but got {type(self._data)}')

        uintxx = 'uint' + str(self._data.itemsize*8)
        A_uint = np.frombuffer(self.data, uintxx)
        self._bitinformation = self.__mutual_information(A_uint, szi=self._szi, confidence=self._confidence)

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, confidence):
        if self._confidence != confidence:
            self._confidence = confidence
            self._parameter_changed = True

    @property
    def szi(self):
        return self._szi

    @szi.setter
    def szi(self, szi):
        if self._szi != szi:
            self._szi = szi
            self._parameter_changed = True

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.__compute_bitinformation()

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
        self._precision = precision

    def bitinformation(self):
        if self._parameter_changed:
            self.__compute_bitinformation()
            self._parameter_changed = False
        return self._bitinformation

    @property
    def mask(self):
        ''' create a mask for removing the least significant zeros,
          e.g., 1111.1111.1000.0000 '''
        nbits = self._data.itemsize * 8
        nbits_used = self.nbits_used
        uintxx = 'uint' + str(nbits)
        T = np.dtype(uintxx).type

        mask = T(0x0)
        for x in range(0, nbits-nbits_used):
            mask = mask << T(0x1) | T(0x1)
        mask = ~mask
        return mask

    @property
    def nbits_used(self):
        ''' create a mask for removing the least significant zeros,
          e.g., 1111.1111.1000.0000 '''
        inf = self.bitinformation()
        nbits = inf.size
        uintxx = 'uint' + str(nbits)
        T = np.dtype(uintxx).type
        total_inf = np.sum(inf)
        tab = pd.DataFrame({'inf':np.flip(inf)})
        tab['cs'] = tab['inf'].cumsum()
        tab = tab.loc[tab['cs'] > (1 - self._precision) * total_inf]
        nbits = tab['cs'].size
        return nbits

    def compare(self, data):
        if self._verbose > 0:
            df = pd.DataFrame({'Value1':data, 'Value2':cleaned_data, 'diff':(data - cleaned_data)}).reset_index()
            df['index'] += 1
            df = df[df['diff'] != 0.0]
            if not df.empty:
                print(df)
        return np.all(self._data == data)
