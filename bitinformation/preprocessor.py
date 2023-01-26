from simple_packing import *

class Preprocessor:
    def __init__(self):
        self._nbits = 0

    def get_numpy_type(nbits):
        if nbits <= 8:
            return np.uint8;
        if nbits <= 16:
            return np.uint16;
        if nbits <= 32:
            return np.uint32;
        if nbits <= 64:
            return np.uint64;

    def convert(self, value):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def bits_per_value(self):
        return self._nbits

class SimplePackingPreprocessor(Preprocessor):
    def __init__(self, nbits, nelems):
        self._nbits = nbits
        self._nelems = nelems

    @property
    def name(self):
        return 'simple'

    def convert(self, values):
        sp = SimplePacking()
        data = sp.encode(values, self._nbits)
        T = Preprocessor.get_numpy_type(self._nbits)
        if self._nelems is not None:
            ret = data[:self._nelems].astype(T)
        else:
            ret = data.astype(T)
        return ret

class ScalePreprocessor(Preprocessor):
    def __init__(self, nbits):
        self._nbits = nbits

    @property
    def name(self):
        return 'scale'

    def convert(self, data):
        min = np.min(data)
        max = np.max(data)
        res = ((data - min)/(max - min) * (2**self._nbits - 1)).astype(np.int16)
        T = Preprocessor.get_numpy_type(self._nbits)
        return res.astype(T)

class RawPreprocessor(Preprocessor):
    def __init__(self):
        self._nbits = -1

    @property
    def name(self):
        return 'raw'

    def convert(self, data):
        self._nbits = data.itemsize * 8
        return data

