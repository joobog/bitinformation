import numpy as np
import pandas as pd

class Table:
    def __init__(self):
        self._tab = list()

    def add(self, entry):
        tab_entry = entry.copy()
        self._tab.append(tab_entry)

    def save(self, fn):
        df = pd.DataFrame(self._tab)
        df.to_csv(fn, index=False)

    def print(self):
        print(self._tab)

class LongTable(Table):
    def add(self, entry, bitinformation):
        for idx, value in zip(np.flip(np.arange(0, bitinformation.size)), bitinformation):
            tab_entry = entry.copy()
            tab_entry['bitpos'] = idx
            tab_entry['information'] = value
            self._tab.append(tab_entry)

class WideTable(Table):
    def add(self, entry, bitinformation):
        tab_entry = entry.copy()
        for idx, value in zip(np.flip(np.arange(0, bitinformation.size)), bitinformation):
            tab_entry[f'b{idx}'] = value
        self._tab.append(tab_entry)
