#!/usr/bin/env python3

import unittest
import numpy as np

from bitinformation.bitinformation import BitInformation

class TestBitInformation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBitInformation, self).__init__(*args, **kwargs)
        self.binfo = BitInformation()


    def test_bitinformation(self):
        A = np.ndarray(
            shape=(10),
            buffer=np.array([12, 14, 17, 19, 20, 18, 15, 12, 11, 10], dtype=np.float64),
            dtype=np.float64
        )
        binfo = BitInformation()
        b = binfo.bitinformation(A)
        a = np.array([
            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0,0,0,
            0,0,0,0
        ])
        self.assertTrue(np.array_equal(a, b))

    def test_permute_dim_forward(self):
        a = np.zeros((3, 4, 5, 6, 7))
        res = self.binfo._BitInformation__permute_dim_forward(a, 3)
        self.assertTrue(np.array_equal((5, 6, 7, 3, 4), res.shape))
        res = self.binfo._BitInformation__permute_dim_forward(a, 4)
        self.assertTrue(np.array_equal((6, 7, 3, 4, 5), res.shape))


    def test_binom_confidence(self):
        c = 0.95
        n = 1000
        res = self.binfo._BitInformation__binom_confidence(n, c)
        self.assertEqual(0.5309897516152281, res)


if __name__ == "__main__":
    unittest.main()
