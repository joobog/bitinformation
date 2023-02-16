#!/usr/bin/env python3

import yaml
import pandas as pd
import numpy as np
from os.path import exists
import copy
import sys
import os
import traceback

class KeyNotFoundException(Exception):
    pass

def compactify(df, key_columns, value_columns):
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
    def __init__(self, filename, key_columns=None, value_columns=None, verbosity=0):
        self.__fn = filename
        self.__key_columns = list(key_columns)
        self.__value_columns = list(value_columns)
        self.__verbosity = verbosity

        if exists(self.__fn):
            with open(self.__fn, 'r') as f:
                y = yaml.safe_load(f)
            df = pd.DataFrame.from_records(y)

            if df.empty:
                raise Exception('Couldn\'t derive primary key from parameter file.')
            self.__data = df
            for key in self.__key_columns:
                try:
                    self.__data = self.__data.explode(key)
                except KeyError as e:
                    print(f'Error: "{key}" is not available in \"{self.__fn}\".')
                    print(f'Data:')
                    print(self.__data)
                    print('Exiting')
                    sys.exit(1)
            self.__data = self.__data.set_index(self.__key_columns)
        else:
            if not key_columns or not value_columns:
                raise Exception('Couldn\'t find a primary key. Please provide a valid parameter file or a primary-key.')
            # self.__data = pd.DataFrame(columns=self.__value_columns+self.__key_columns)
            self.__data = pd.DataFrame()
            # try:
            #     self.__data = df.set_index(self.__key_columns)
            # except NotImplementedError as e:
            #     # exc_type, exc_obj, exc_tb = sys.exc_info()
            #     # fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            #     # print(exc_type, fname, exc_tb.tb_lineno)
            #     print(f'Error: Parameters table is empty. Cannot set index {self.__key_columns}')
            #     print('Exiting')
            #     # sys.exit(1)
            #     raise

    def __select_keys_as_tuple(self, unordered_dict_index, keys):
        '''Helper function: get dict names'''
        try:
            return tuple(unordered_dict_index[key] for key in keys)
        except KeyError as e:
            # traceback.print_exc()
            print(f'Error: Couldn\'t find key: "{e}"')
            print(f'Available keys are:')
            for key in unordered_dict_index.keys():
                print(f'\t{key}')
            print(f'Exiting')
            raise


    def __select_keys_as_dict(self, unordered_dict_index, keys):
        '''Helper function: get dict names'''
        try:
            return {key:unordered_dict_index[key] for key in keys}
        except KeyError as e:
            # traceback.print_exc()
            print(f'Error: Couldn\'t find key: "{e}"')
            print(f'Available keys are:')
            for key in unordered_dict_index.keys():
                print(f'\t{key}')
            print(f'Exiting')
            raise

    def save(self):
        if not self.__data.empty:
            # print(self.__data)
            compact_data = compactify(self.__data, key_columns=self.__key_columns, value_columns=self.__value_columns)
            yaml_data = compact_data.to_dict(orient='records')
            with open(self.__fn, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=None)
        else:
            if self.__verbosity >= 1:
                print('Warning: Nothing to save. Skipping.')

    def add(self, metadata):
        key = self.__select_keys_as_tuple(metadata, self.__key_columns)
        value = self.__select_keys_as_dict(metadata, self.__value_columns)
        if key in self.__data.index:
            for vname in self.__value_columns:
                self.__data.loc[key][vname] = value[vname]
        else:
            entry = self.__select_keys_as_dict(metadata, self.__key_columns+self.__value_columns)
            row = pd.DataFrame({k:[v] for k,v in entry.items()}).set_index(self.__key_columns)
            self.__data = pd.concat([self.__data, row])

    def lookup(self, key):
        key_tuple = self.__select_keys_as_tuple(key, self.__key_columns)
        return self.__data.loc[key_tuple].to_dict()

    def key_exists(self, metadata):
        key = self.__select_keys_as_tuple(metadata, self.__key_columns)
        return key in self.__data.index

    def print(self):
        print(self.__data)

    def get_key_columns(self):
        return self.__key_columns

    def get_value_columns(self):
        return self.__value_columns
