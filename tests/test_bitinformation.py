#!/usr/bin/env python3

import unittest
import numpy as np
from eccodes import *

from bitinformation.bitinformation import BitInformation
# from bitinformation import BitInformation

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

    def test_compare_with_julia(self):
        # q_133.grib
        output_julia_q_0_4 = np.array(
            [0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 1.25201478e-01, 1.25201478e-01, 1.25201478e-01,
             3.38865801e-01, 8.52290511e-01, 8.54100732e-01, 8.12767209e-01,
             6.89931434e-01, 5.18894663e-01, 3.18227831e-01, 1.41773901e-01,
             3.40563637e-02, 2.84270695e-03, 7.52561783e-05, 2.00566853e-06,
             2.68482222e-07, 1.16855302e-07, 0.00000000e+00, 2.32399475e-02,
             0.00000000e+00, 1.25201478e-01, 0.00000000e+00, 0.00000000e+00,
             1.25201478e-01, 1.25201478e-01, 1.25201478e-01, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00,
             0.00000000e+00, 0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
            dtype=np.float64)

        output_julia_q_0_99 = np.array(
            [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,1.25201478e-01,1.25201478e-01,1.25201478e-01,
             3.38865801e-01,8.52290511e-01,8.54100732e-01,8.12767209e-01,
             6.89931434e-01,5.18894663e-01,3.18227831e-01,1.41773901e-01,
             3.40563637e-02,2.84270695e-03,7.52561783e-05,2.00566853e-06,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,2.32399475e-02,
             0.00000000e+00,1.25201478e-01,0.00000000e+00,0.00000000e+00,
             1.25201478e-01,1.25201478e-01,1.25201478e-01,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
            dtype=np.float64)

        output_julia_q_1 = np.array(
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            dtype=np.float64)

        with open('data/q_133.grib') as f:
            handle = codes_grib_new_from_file(f)
            values = codes_get_values(handle)
            codes_release(handle)

            output_python = self.binfo.bitinformation(values, szi=True, confidence=0.4)
            np.testing.assert_allclose(output_python, output_julia_q_0_4, atol=1e-06)
            output_python = self.binfo.bitinformation(values, szi=True, confidence=0.99)
            np.testing.assert_allclose(output_python, output_julia_q_0_99, atol=1e-06)
            output_python = self.binfo.bitinformation(values, szi=True, confidence=1.)
            np.testing.assert_allclose(output_python, output_julia_q_1, atol=1e-06)


        # o3_133.grib
        output_julia_o3_0_4 = np.array(
            [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,8.84313907e-01,
             8.76210837e-01,7.85047259e-01,6.38733573e-01,4.80014798e-01,
             2.80295029e-01,1.24641227e-01,3.92977885e-02,7.99398724e-03,
             1.08036295e-03,1.18707970e-04,8.05443395e-06,5.25753558e-07,
             1.84420178e-07,0.00000000e+00,8.17680426e-02,8.84313907e-01,
             0.00000000e+00,8.84313907e-01,8.84313907e-01,8.84313907e-01,
             8.84313907e-01,8.84313907e-01,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
            dtype=np.float64)

        output_julia_o3_0_99 = np.array(
            [0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,8.84313907e-01,
             8.76210837e-01,7.85047259e-01,6.38733573e-01,4.80014798e-01,
             2.80295029e-01,1.24641227e-01,3.92977885e-02,7.99398724e-03,
             1.08036295e-03,1.18707970e-04,8.05443395e-06,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,8.17680426e-02,8.84313907e-01,
             0.00000000e+00,8.84313907e-01,8.84313907e-01,8.84313907e-01,
             8.84313907e-01,8.84313907e-01,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00,
             0.00000000e+00,0.00000000e+00,0.00000000e+00,0.00000000e+00],
            dtype=np.float64)

        output_julia_o3_1 = np.array(
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            dtype=np.float64)

        with open('data/o3_203.grib') as f:
            handle = codes_grib_new_from_file(f)
            values = codes_get_values(handle)
            codes_release(handle)

            output_python = self.binfo.bitinformation(values, szi=True, confidence=0.4)
            np.testing.assert_allclose(output_python, output_julia_o3_0_4, atol=1e-06)
            output_python = self.binfo.bitinformation(values, szi=True, confidence=0.99)
            np.testing.assert_allclose(output_python, output_julia_o3_0_99, atol=1e-06)
            output_python = self.binfo.bitinformation(values, szi=True, confidence=1.)
            np.testing.assert_allclose(output_python, output_julia_o3_1, atol=1e-06)


        # clwc_246.grib
        output_julia_clwc_0_4 = np.array(
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            dtype=np.float64)

        output_julia_clwc_0_99 = np.array(
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            dtype=np.float64)

        output_julia_clwc_1 = np.array(
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            dtype=np.float64)

        with open('data/clwc_246.grib') as f:
            handle = codes_grib_new_from_file(f)
            values = codes_get_values(handle)
            codes_release(handle)

            output_python = self.binfo.bitinformation(values, szi=True, confidence=0.4)
            np.testing.assert_allclose(output_python, output_julia_clwc_0_4, atol=1e-06)
            output_python = self.binfo.bitinformation(values, szi=True, confidence=0.99)
            np.testing.assert_allclose(output_python, output_julia_clwc_0_99, atol=1e-06)
            output_python = self.binfo.bitinformation(values, szi=True, confidence=1.)
            np.testing.assert_allclose(output_python, output_julia_clwc_1, atol=1e-06)


        # asn_32.grib
        output_julia_ans_0_4 = np.array(
            [0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.10781152,0.08510378,0.12134845,0.21098934,0.18573434,0.059191,
             0.05756693,0.05244674,0.15151083,0.04981514,0.15056503,0.14981766,
             0.04937336,0.14898387,0.14946056,0.04834458,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,],
            dtype=np.float64)

        output_julia_ans_0_99 = np.array(
            [0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.10781152,0.08510378,0.12134845,0.21098934,0.18573434,0.059191,
             0.05756693,0.05244674,0.15151083,0.04981514,0.15056503,0.14981766,
             0.04937336,0.14898387,0.14946056,0.04834458,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,],
            dtype=np.float64)

        output_julia_ans_1 = np.array(
            [0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,
             0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
            dtype=np.float64)

        with open('data/asn_32.grib') as f:
            handle = codes_grib_new_from_file(f)
            values = codes_get_values(handle)
            codes_release(handle)

            output_python = self.binfo.bitinformation(values, szi=True, confidence=0.4)
            np.testing.assert_allclose(output_python, output_julia_ans_0_4, atol=1e-06)
            output_python = self.binfo.bitinformation(values, szi=True, confidence=0.99)
            np.testing.assert_allclose(output_python, output_julia_ans_0_99, atol=1e-06)
            output_python = self.binfo.bitinformation(values, szi=True, confidence=1.)
            np.testing.assert_allclose(output_python, output_julia_ans_1, atol=1e-06)

if __name__ == "__main__":
    unittest.main()
