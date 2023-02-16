
#!/usr/bin/env python3

import unittest
import numpy as np

from bitinformation.simple_packing import SimplePacking

class TestBitInformation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBitInformation, self).__init__(*args, **kwargs)

    def test_eccodes_output(self):
        sp = SimplePacking()

        a = np.array([3, 10, 5], dtype=np.float64)
        _, _, _, encoded = sp.encode(a, 16)
        expected = np.array([0, 57344, 16384], dtype=np.uint64)
        np.testing.assert_array_equal(encoded, expected)

        # a = np.array([0.000001, 0.12000003, 0.0094198888890023, 10.0000000001], dtype=np.float64)
        # expected = np.array([0, 492, 39, 40960], dtype=np.uint64)

        np.testing.assert_array_equal(encoded, expected)


    # def test_eccodes_expr(self):
    #     sp = SimplePacking()
    #     a = np.array([0.123, 0.124, 0.125], dtype=np.float64)
    #     R, E, D, encoded = sp.encode(a, 24)
    #     print('ENCODED ', R, E, D, np.array([17, 4294985, 8589952]))
    #     decoded = sp.decode(R, E, D, encoded)
    #     print('DECODED ', decoded)
    #     decoded = sp.decode(R, E, D, np.array([17, 4294980, 8589950]))
    #     print('DECODED ', decoded)
    #     decoded = sp.decode(R, E, D, np.array([17, 4294900, 8589900]))
    #     print('DECODED ', decoded)
    #     decoded = sp.decode(R, E, D, np.array([17, 4294000, 8589000]))
    #     print('DECODED ', decoded)
    #     decoded = sp.decode(R, E, D, np.array([17, 4290000, 8580000]))
    #     print('DECODED ', decoded)
    #     decoded = sp.decode(R, E, D, np.array([17, 4200000, 8500000]))
    #     print('DECODED ', decoded)
    #     decoded = sp.decode(R, E, D, np.array([1, 4000000, 8000000]))
    #     print('DECODED ', decoded)
    #     assert(False)

if __name__ == "__main__":
    unittest.main()

# TODO: make test
# if '__main__' == __name__:
#     bits_per_value = 16
#     values = np.array([0.12, 0.23, 0.42, 0.54, 0.12, 0.30, 0.12], dtype=np.float64)
#     sp = SimplePacking()
#     data = sp.encode(values, bits_per_value)
#     print(data)
