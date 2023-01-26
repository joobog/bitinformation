#!/usr/bin/env python3

import sys
sys.path.append('./bitinformation')
import unittest
import numpy as np
from eccodes import *

from tool import *
from analyser import *
from preprocessor import *

class TestTool(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTool, self).__init__(*args, **kwargs)

    def test_get_stats(self):
        tool = Tool()

        filename = './data/q_133.grib'

        preprocessor = SimplePackingPreprocessor(nbits=16, nelems=10000)
        analyser = BitInformationAnalyser(szi=True, confidence=0.99, precision=0.99)

        with open(filename, 'rb') as f:
            handle = codes_grib_new_from_file(f)
            metadata, info = tool.stats(handle, preprocessor, analyser)
            del metadata['timestamp']
            print(metadata)
            print(info.bitinformation)
            codes_release(handle)
            metadata_expected = {'nbits_total': 16, 'nbits_value': 16, 'nbits_used': 8, 'preprocessor': 'simple', 'const': 0, 'short_name': 'q', 'domain': 'g', 'date': '20221106', 'time': '0000', 'expver': '0001', 'class': 'od', 'type': 'an', 'stream': 'oper', 'step': '0', 'levtype': 'ml', 'levelist': '1', 'param': '133'}
            bitinformation_expected = [
                0.        , 0.18605277, 0.63944644, 0.57452212, 0.48286549, 0.26685169,
                0.09141606, 0.04287478, 0.00686294, 0.00181347, 0.        , 0.,
                0.        , 0.        , 0.        , 0.]

            self.assertTrue(metadata_expected == metadata)
            np.testing.assert_allclose(bitinformation_expected, info.bitinformation, atol=1e-06)

    def test_get_stats(self):
        tool = Tool()
        fn1 = './data/q_133.grib'
        fn2 = './data/q_133.grib'

        preprocessor = SimplePackingPreprocessor(nbits=16, nelems=10000)
        analyser = BitInformationAnalyser(szi=True, confidence=0.99, precision=0.99)

        f1 = open(fn1, 'rb')
        f2 = open(fn2, 'rb')
        handle1 = codes_grib_new_from_file(f1)
        handle2 = codes_grib_new_from_file(f2)
        res = tool.compare(handle1, handle2, preprocessor=preprocessor, analyser=analyser)
        self.assertTrue(res)
        codes_release(handle1)
        codes_release(handle2)
        f1.close()
        f2.close()

if __name__ == "__main__":
    unittest.main()
