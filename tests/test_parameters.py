#!/usr/bin/env python3

import unittest
import os
import pandas as pd
from pandas.testing import assert_frame_equal

from bitinformation.parameters import Parameters

class TestParameters(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestParameters, self).__init__(*args, **kwargs)
        self.__fn = 'test.yaml'

    def setUp(self):
        self.__params = Parameters(
            self.__fn,
            key_columns=('param_id', 'precision', 'stream', 'levelist'),
            value_columns=('nbits_used', 'nbits')
        )

    def tearDown(self):
        if os.path.exists(self.__fn):
            os.remove(self.__fn)

    def test_empty_file(self):
        self.__params.print()
        self.__params.save()

    def test_add_parameter(self):
        self.__params.add(metadata={'param_id':'101', 'precision':'0.99', 'stream':'oper', 'levelist':'124', 'nbits_used':'4', 'nbits':'24'})
        self.__params.add(metadata={'param_id':'101', 'precision':'0.99', 'stream':'oper', 'levelist':'127', 'nbits_used':'4', 'nbits':'24'})
        df = pd.DataFrame({'param_id':['101', '101'], 'precision':['0.99', '0.99'], 'stream':['oper', 'oper'], 'levelist':['124', '127'], 'nbits_used':['4', '4'], 'nbits':['24', '24']})
        df = df.set_index(['param_id', 'precision', 'stream', 'levelist'])
        assert_frame_equal(df.sort_index(axis=1), self.__params._Parameters__data.sort_index(axis=1))

    def test_update_parameter(self):
        self.__params.add(metadata={'param_id':'101', 'precision':'0.99', 'stream':'oper', 'levelist':'124', 'nbits_used':'4', 'nbits':'24'})
        self.__params.add(metadata={'param_id':'101', 'precision':'0.99', 'stream':'oper', 'levelist':'127', 'nbits_used':'4', 'nbits':'24'})
        self.__params.add(metadata={'param_id':'101', 'precision':'0.99', 'stream':'oper', 'levelist':'127', 'nbits_used':'5', 'nbits':'24'})
        df = pd.DataFrame({'param_id':['101', '101'], 'precision':['0.99', '0.99'], 'stream':['oper', 'oper'], 'levelist':['124', '127'], 'nbits_used':['4', '5'], 'nbits':['24', '24']})
        df = df.set_index(['param_id', 'precision', 'stream', 'levelist'])
        assert_frame_equal(df.sort_index(axis=1), self.__params._Parameters__data.sort_index(axis=1))

    def test_load(self):
        self.__params.add(metadata={'param_id':'101', 'precision':'0.99', 'stream':'oper', 'levelist':'124', 'nbits_used':'4', 'nbits':'24'})
        self.__params.add(metadata={'param_id':'101', 'precision':'0.99', 'stream':'oper', 'levelist':'127', 'nbits_used':'5', 'nbits':'24'})
        self.__params.save()
        params = Parameters(
            self.__fn,
            key_columns=('param_id', 'precision', 'stream', 'levelist'),
            value_columns=('nbits_used', 'nbits')
        )
        df = pd.DataFrame({'param_id':['101', '101'], 'precision':['0.99', '0.99'], 'stream':['oper', 'oper'], 'levelist':['124', '127'], 'nbits_used':['4', '5'], 'nbits':['24', '24']})
        df = df.set_index(['param_id', 'precision', 'stream', 'levelist'])
        assert_frame_equal(df.sort_index(axis=1), params._Parameters__data.sort_index(axis=1))

if __name__ == "__main__":
    unittest.main()
