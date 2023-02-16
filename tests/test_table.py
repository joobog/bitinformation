#!/usr/bin/env python3

import unittest
import numpy as np

from bitinformation.table import *

class TestTable(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super(TestTable, self).__init__(*args, **kwargs)

    def test_abstract_class(self):
        tab = Table()
        tab.add({'a':'x', 'b': 'y'})
        expected = [{'a':'x', 'b': 'y'}]
        self.assertTrue(tab._tab == expected)

    def test_long_format(self):
        entry = {'a':'x', 'b': 'y'}
        bitinformation = np.array([0, 1, 2, 3])
        tab = LongTable()
        tab.add(entry, bitinformation)

        expected = [
            {'a': 'x', 'b': 'y', 'bitpos': 3, 'information': 0},
            {'a': 'x', 'b': 'y', 'bitpos': 2, 'information': 1},
            {'a': 'x', 'b': 'y', 'bitpos': 1, 'information': 2},
            {'a': 'x', 'b': 'y', 'bitpos': 0, 'information': 3}
        ]

        self.assertTrue(tab._tab == expected)

    def test_wide_format(self):
        entry = {'a':'x', 'b': 'y'}
        bitinformation = np.array([0, 1, 2, 3])
        tab = WideTable()
        tab.add(entry, bitinformation)

        expected = [
            {'a': 'x', 'b': 'y', 'b3':0, 'b2':1, 'b1': 2, 'b0': 3},
        ]

        print(tab._tab)
        self.assertTrue(tab._tab == expected)

if __name__ == "__main__":
    unittest.main()
