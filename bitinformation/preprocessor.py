import numpy as np
from simple_packing import SimplePacking, ConstantFieldException

class Preprocessor:
    def __init__(self):
        self._name = None
        self._bits_per_value = None
        self._T = None
        self._nelems = None

    def encode(self, data):
        raise NotImplementedError

    def decode(self, data, params):
        raise NotImplementedError

    def name(self):
        return self._name

    def bits_per_value(self):
        return self._bits_per_value

    def type(self):
        return self._T

    def metadata(self):
        return {
            'name': self.name(),
            'bits_per_value': self._bits_per_value,
            'nelems': self._nelems,
            'itemsize': np.dtype(self._T).itemsize * 8
        }

    @staticmethod
    def get_optimal_numpy_type(bits_per_value):
        assert bits_per_value >= 1
        assert bits_per_value <= 64
        if bits_per_value <= 8:
            T = np.uint8
        elif bits_per_value <= 16:
            T = np.uint16
        elif bits_per_value <= 32:
            T = np.uint32
        elif bits_per_value <= 64:
            T = np.uint64
        else:
            raise Exception(f'Could not find an optimal type for {bits_per_value} bits.')
        return T

class SimplePackingPreprocessor(Preprocessor):
    def __init__(self, bits_per_value, nelems):
        super(SimplePackingPreprocessor, self).__init__()
        self.__sp = SimplePacking()
        self._bits_per_value = bits_per_value
        self._nelems = nelems
        self._T = Preprocessor.get_optimal_numpy_type(self._bits_per_value)
        self._name = 'simple'

    def encode(self, data):
        params = dict()
        R, E, D, data = self.__sp.encode(data, self._bits_per_value)
        params['reference'] = R
        params['binary_scale'] = E
        params['decimal_scale'] = D
        if self._nelems is None:
            ret = data.astype(self._T)
        else:
            ret = data[:self._nelems].astype(self._T)
        return (ret, params)

    def decode(self, data, params):
        return self.__sp.decode(params['referece'], params['binary_scale'], params['decimal_scale'], data)

class ScalePreprocessor(Preprocessor):
    def __init__(self, bits_per_value, nelems):
        super(ScalePreprocessor, self).__init__()
        self._bits_per_value = bits_per_value
        self._nelems = nelems
        self._T = Preprocessor.get_optimal_numpy_type(self._bits_per_value)
        self._name = 'scale'

    def encode(self, data):
        vmin = np.min(data)
        vmax = np.max(data)
        res = ((data - vmin)/(vmax - vmin) * (2**self._bits_per_value - 1)).astype(np.int16)
        T = Preprocessor.get_optimal_numpy_type(self._bits_per_value)
        decoding_parameters = dict()
        if self._nelems is not None:
            return res[:self._nelems].astype(T)
        return (res.astype(T), decoding_parameters)

    def decode(self, data, params):
        raise NotImplementedError

class RawPreprocessor(Preprocessor):
    def __init__(self, bits_per_value, nelems):
        super(RawPreprocessor, self).__init__()
        self._bits_per_value = bits_per_value
        self._nelems = nelems
        self._T = Preprocessor.get_optimal_numpy_type(self._bits_per_value)
        self._name = 'raw'

    def encode(self, data):
        assert data.itemsize * 8 == self._bits_per_value
        assert data.itemsize == np.dtype(self._T).itemsize
        if self._nelems is not None:
            return data[:self._nelems]
        decoding_parameters = dict()
        return (data, decoding_parameters)

    def decode(self, data, params):
        raise NotImplementedError
