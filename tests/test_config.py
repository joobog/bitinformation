#!/usr/bin/env python3

import unittest
from bitinformation.config import *

class TestBitInformation(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestBitInformation, self).__init__(*args, **kwargs)

    def test_config_lookup(self):
        config = Config('config.yml')
        config.add(1   , 'oper' , 8, 0.99)
        config.add(2   , 'oper' , 8, 0.99)
        config.add(3   , 'oper' , 5, 0.99)
        config.add(3   , 'oper' , 9, 0.99, replace=True)
        config.add(4   , 'st'   , 7, 0.99)
        config.add(5   , 'st'   , 7, 0.99)
        config.add(6   , 'st'   , 6, 0.99)
        config.add(200 , 'st'   , 6, 0.99)

        res = config.lookup(paramid=200, stream='st', precision=0.99)
        assert(res['nbits'] == 6)
        res = config.lookup(paramid=3, stream='oper', precision=0.99)
        assert(res['nbits'] == 9)

if __name__ == "__main__":
    unittest.main()

