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
    entropy = 1 - scipy.stats.entropy([p, 1-p], base=base)
    return entropy

class Analyser:
    # class Result:
    #     def __init__(self,
    #                  bitinformation,
    #                  nbits_used,
    #                  masked_data,
    #                  mask,
    #                  equal):
    #         self._bitinformation = bitinformation
    #         self._nbits_used = nbits_used
    #         self._masked_data = masked_data
    #         self._mask = mask
    #         self._equal = equal

    #     @property
    #     def bitinformation(self):
    #         return self._bitinformation

    #     @property
    #     def nbits_used(self):
    #         return self._nbits_used

    #     @property
    #     def masked_data(self):
    #         return self._masked_data

    #     @property
    #     def mask(self):
    #         return self._mask

    #     @property
    #     def equal(self):
    #         return self._equal

    #     def print(self):
    #         print(f'Bitinformation: \n{self._bitinformation}')
    #         print(f'nbits_used: {self._nbits_used}')
    #         print(f'mask:       {bin(self._mask)}')
    #         print(f'equal:      {self._equal}')

    def analyse(self, data):
        raise NotImplementedError

    def metadata(self):
        raise NotImplementedError

    def name(self):
        raise NotImplementedError

    @staticmethod
    def masked_data(data, mask, verbose=0):
        OT = data.dtype.type
        uintxx = 'uint' + str(data.itemsize*8)
        data_uint = np.frombuffer(data, uintxx)
        if verbose >= 1:
            print('Used bit mask: ', bin(mask))
        a = data_uint & mask
        return np.frombuffer(a, OT)

    @staticmethod
    def compare(data, masked_data, verbose=0):
        if verbose > 0:
            df = pd.DataFrame({'Value1':data, 'Value2':masked_data, 'diff':(data - masked_data)}).reset_index()
            df['index'] += 1
            df = df[df['diff'] != 0.0]
            if not df.empty:
                print(df)
        return np.all(data == masked_data)



class MaskAnalyser(Analyser):
    def __init__(self, mask, verbose=0):
        self._mask = mask
        self._verbose = verbose

    @property
    def mask(self):
        return self._mask

    @mask.setter
    def mask(self, mask):
        self._mask = mask

    def __bitinformation(self, data):
        U = data.dtype.type
        mask = self._mask
        nbits = data.itemsize * 8
        bi = np.zeros(nbits)
        for i in np.flip(np.arange(start=0, stop=nbits)):
            bi[i] = mask & U(0x1)
            mask = mask >> U(0x1)
        return bi

    def analyse(self, data):
        bitinformation = self.__bitinformation(data)
        nbits_used = np.sum(bitinformation.astype(np.int64))
        masked_data = Analyser.masked_data(data, self._mask)
        mask = self._mask
        equal = Analyser.compare(data, masked_data)
        return {
            'bitinformation': bitinformation,
            'nbits_used': nbits_used,
            'masked_data': masked_data,
            'mask': mask,
            'equal': equal
        }
        # return Analyser.Result(bitinformation, nbits_used, masked_data, mask, equal)

    def name(self):
        return 'mask'

    def metadata(self):
        return {
            'name': self.name(),
            'mask': self._mask
        }


class BitInformationAnalyser(Analyser):
    def __init__(self, szi=True, confidence=0.99, precision=0.99, verbosity=0):
        self.szi = szi
        self.confidence = confidence
        self.precision = precision
        self._verbose = verbosity

    def name(self):
        return 'mask'

    def metadata(self):
        return {
            'name': self.name(),
            'szi': self.szi,
            'confidence': self.confidence,
            'precision': self.precision
        }

    @property
    def confidence(self):
        return self._confidence

    @confidence.setter
    def confidence(self, confidence):
        assert(confidence >= 0 and confidence <= 1)
        self._confidence = confidence

    @property
    def szi(self):
        return self._szi

    @szi.setter
    def szi(self, szi):
        self._szi = szi

    @property
    def verbose(self):
        return self._verbose

    @verbose.setter
    def verbose(self, verbose):
        self._verbose = verbose

    @property
    def precision(self):
        return self._precision

    @precision.setter
    def precision(self, precision):
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
        C = np.zeros((nbits, 2, 2), dtype=np.uint64)
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
                if p[i, j] > 0:
                    M += p[i, j] * np.log(p[i, j] / px[i] / py[j])
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
        P = np.zeros((2, 2))
        for i in range(0, nbits):
            for j in [0, 1]:
                for k in [0, 1]:
                    P[j, k] = C[i, j, k] / nelements
            M[i] = self.__mutual_information2(P)

        # remove information that is insignificantly different from a random 50/50 experiment
        if szi:
            self.__szi(M, nelements, confidence)
        return M

    def __bitinformation(self, data):
        if not isinstance(data, np.ndarray):
            raise Exception(f'Expect numpy.ndarray as parameter but got {type(data)}')

        uintxx = 'uint' + str(data.itemsize*8)
        A_uint = np.frombuffer(data, uintxx)
        bitinformation = self.__mutual_information(A_uint, szi=self._szi, confidence=self._confidence)
        return bitinformation

    def __mask(self, data, nbits_used):
        ''' create a mask for removing the least significant zeros,
          e.g., 1111.1111.1000.0000 '''
        nbits = data.itemsize * 8
        uintxx = 'uint' + str(nbits)
        T = np.dtype(uintxx).type

        mask = T(0x0)
        for _ in range(0, nbits-nbits_used):
            mask = mask << T(0x1) | T(0x1)
        mask = ~mask
        return mask

    def __nbits_used(self, inf):
        total_inf = np.sum(inf)
        tab = pd.DataFrame({'inf':np.flip(inf)})
        tab['cs'] = tab['inf'].cumsum()
        tab = tab.loc[tab['cs'] > (1 - self._precision) * total_inf]
        nbits_used = tab['cs'].size

        # The data type can contain more bits than the analysed numbers
        # e.g. uint32 can contain a number with 24 bits per value
        # nbits_used = nbits_used - (nbits - self._nbits_avail)
        # print("NBITS", nbits_used)
        # print(inf)
        return nbits_used

    def analyse(self, data):
        bitinformation = self.__bitinformation(data)
        nbits_used = self.__nbits_used(bitinformation)
        mask = self.__mask(data, nbits_used)
        masked_data = Analyser.masked_data(data=data, mask=mask)
        equal = Analyser.compare(data, masked_data)
        return {
            'bitinformation': bitinformation,
            'nbits_used': nbits_used,
            'masked_data': masked_data,
            'mask': mask,
            'equal': equal
        }
        # return self.Result(bitinformation, nbits_used, masked_data, mask, equal)
