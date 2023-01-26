#!/usr/bin/env python3

import yaml
import pandas as pd
import numpy as np
from os.path import exists
import copy
import sys
import traceback


def compactify(df, value_columns, key_columns):
    ''' Creates a compact representation of the parameter table'''
    compact_df = df.reset_index()
    for i in range(0, len(key_columns)):
        ck = copy.deepcopy(key_columns)
        del ck[i]
        keys = value_columns + ck
        compact_df = compact_df.groupby(keys).agg(tuple).reset_index()
    for compact_key in key_columns:
        compact_df[compact_key] = compact_df[compact_key].apply(list)
    return compact_df

class Parameters:
    ''' Read and write parameters'''
    def __init__(self, filename, key_columns=None, value_columns=None):
        # Support only one value column
        assert(len(value_columns) == 1)
        self.__fn = filename
        self.__key_columns = list(key_columns)
        self.__value_columns = list(value_columns)

        if exists(self.__fn):
            with open(self.__fn, 'r') as f:
                y = yaml.safe_load(f)
            df = pd.DataFrame.from_records(y)
            if df.empty:
                raise Exception('Couldn\'t derive primary key from parameter file.')
        else:
            if not key_columns or not value_columns:
                raise Exception('Couldn\'t derive primary key. Please provide a valid parameter file or a primary-key.')
            else:
                df = pd.DataFrame(columns=self.__value_columns+self.__key_columns)
        self.__data = df
        for compact_key in self.__key_columns:
            try:
                self.__data = self.__data.explode(compact_key)
            except Exception as e:
                print(f'Error: Incompatible parameter file found {self.__fn}: "{compact_key}" is not available.')
                sys.exit(1)
        self.__data = self.__data.set_index(self.__key_columns)

    def __dict_names_to_tuple(self, unordered_dict_index):
         '''Helper function: get dict names'''
         try:
             return tuple(unordered_dict_index[colname] for colname in self.__key_columns)
         except KeyError as e:
             traceback.print_exc()
             print(f'Couldn\'t find key: "{e}"')
             print(f'Available keys are {unordered_dict_index}.\nExiting')
             sys.exit(1)

    def save(self):
        compact_data = compactify(self.__data, value_columns=self.__value_columns, key_columns=self.__key_columns)
        yaml_data = compact_data.to_dict(orient='records')
        with open(self.__fn, 'w') as f:
            yaml.dump(yaml_data, f, default_flow_style=None)

    def add(self, metadata, value):
        key = self.__dict_names_to_tuple(metadata)
        if key in self.__data.index:
            self.__data.loc[key] = (value[self.__value_columns[0]], )
        else:
            row = pd.DataFrame({self.__value_columns[0]:value[self.__value_columns[0]]}, index=[key])
            self.__data = pd.concat([self.__data, row])

    def lookup(self, key):
        key = self.__dict_names_to_tuple(key)
        return self.__data.loc[key]

    def print(self):
        print(self.__data)

    def get_key_columns(self):
        return self.__key_columns


if __name__ == '__main__':
    params = Parameters('test.yml', key_columns=('param_id', 'precision', 'stream', 'levelist'), value_columns=('nbits',))
    params.add(key={'param_id':'101', 'precision':'0.99', 'stream':'oper', 'levelist':'124'}, value={'nbits':'4'})
    params.add(key={'param_id':'103', 'precision':'0.99', 'stream':'oper', 'levelist':'127'}, value={'nbits':'4'} )
    params.add(key={'param_id':'103', 'precision':'0.99', 'stream':'oper', 'levelist':'127'}, value={'nbits':'5'}, update=True)
    params.print()
    value = params.lookup({'param_id':'101', 'precision':'0.99', 'stream':'oper', 'levelist':'124'})
    print(value)
    params.save()
