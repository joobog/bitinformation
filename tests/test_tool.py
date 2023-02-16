#!/usr/bin/env python3

import sys
# sys.path.append('./bitinformation')
sys.path.insert(0, './bitinformation')
import unittest
import numpy as np
from eccodes import *
from timeit import default_timer as timer

from tool import *
from analyser import *
from preprocessor import *

class TestTool(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTool, self).__init__(*args, **kwargs)

    def test_compare(self):
        tool = Tool()
        preprocessor = SimplePackingPreprocessor(bits_per_value=16, nelems=10000)
        md1 = {'nbits_used': 5, 'preprocessor_bits_per_value': 16}
        md2 = {'nbits_used': 5, 'preprocessor_bits_per_value': 16}
        values1 = np.array([1, 2, 3])
        values2 = np.array([1, 2, 3])
        result = tool.compare(values1, values2, md1, md2, preprocessor)
        self.assertTrue(result['status'])

if __name__ == "__main__":
    unittest.main()
