
#!/usr/bin/env python3

import unittest
import numpy as np

from bitinformation.simple_packing import *

class TestBitInformation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBitInformation, self).__init__(*args, **kwargs)

    def test_eccodes_output(self):
        sp = SimplePacking()

        a = np.array([3, 10, 5], dtype=np.float64)
        actual = sp.encode(a, 16)
        expected = np.array([0, 57344, 16384], dtype=np.uint64)
        np.testing.assert_array_equal(actual, expected)

        a = np.array([0.000001, 0.12000003, 0.0094198888890023, 10.0000000001], dtype=np.float64)
        actual = sp.encode(a, 16)
        expected = np.array([0, 492, 39, 40960], dtype=np.uint64)
        np.testing.assert_array_equal(actual, expected)

if __name__ == "__main__":
    unittest.main()
