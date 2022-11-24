#!/usr/bin/env python3

from eccodes import *
from simple_packing import *
import bitinformation as bit
import numpy as np
import pandas as pd
import argparse
import sys
import struct
import pathlib


def simple_pack(values, nbits=16):
    ''' Simple packing as used in CCSDS/AEC
    Y*10^D = R+X*2^E '''
    sp = SimplePacking()
    data = sp.encode(values, nbits)
    return data


def scale_pack(data, nbits=16):
    min = np.min(data)
    max = np.max(data)
    res = ((data - min)/(max - min) * (2**16 - 1)).astype(np.int16)
    return res


class BitInfoStats:
    def __init__(self, data, szi=True, confidence=0.99, verbose=0):
        self._inf = bit.BitInformation(data, verbose=verbose, szi=szi, confidence=confidence)
        self._verbose = verbose


    def analyse_data(self):
        inf = self._inf.bitinformation()
        cleaned_data = self._inf.cleaned_data()
        if self._verbose > 0:
            df = pd.DataFrame({'Value1':data, 'Value2':cleaned_data, 'diff':(data - cleaned_data)}).reset_index()
            df['index'] += 1
            df = df[df['diff'] != 0.0]
            if not df.empty:
                print(df)
        is_equal = np.all(data == cleaned_data)

        res = dict()
        res['nbits'] = inf.size
        res['nbits_used'] = self._inf.nbits_used
        res['equal'] = is_equal
        for idx, value in zip(np.arange(0, inf.size), inf):
            res[f'bit_{idx}'] = value
        return res



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('pos', nargs='+', help='command args ...', type=str)
    parser.add_argument('-n', '--number', help='Test the first n values only', type=int)
    parser.add_argument('-c', '--confidence', help='Confidence', type=float, default=0.99)
    parser.add_argument('-z', '--szi', help='Disable set zero insignificant', action="store_true")
    parser.add_argument('-s', '--silent', help='Disable output', action="store_true")
    parser.add_argument('-v', '--verbosity', help='Verbosity level [0|1|2]', type=int, default=1)
    parser.add_argument('-b', '--n_msb', help='Number of most significant bits used for comparision', type=int)
    args = parser.parse_args()
    if args.silent:
        verbose = arg
    verbose = args.verbosity

    if args.verbosity > 2:
        print(args)

    tab = list()
    for fn in args.pos:
        ext = (pathlib.Path(fn)).suffix
        print(f'Processing {fn}')
        if ext == ".csv":
            data = pd.read_csv(fn, nrows=args.number)
            bit_stats = BitInfoStats(data, szi=not args.szi, confidence=args.confidence, verbose=verbose)
            tab_entry = bit_stats.analyse_data(data)
            tab.append(tab_entry)
        elif ext == ".grib":
            f = open(fn, 'r')
            exit_status = 0
            msg_count = 0;
            while True:
                handle = codes_grib_new_from_file(f)
                if not handle:
                    break
                values = codes_get_values(handle)
                short_name = codes_get_string(handle, 'shortName')
                try:
                    data = simple_pack(values)
                except ConstantFieldException:
                    data = np.zeros(values.size, dtype=np.int16)
                bit_stats = BitInfoStats(data, szi=not args.szi, confidence=args.confidence, verbose=verbose)
                tab_entry = dict()
                tab_entry['short_name'] = short_name
                tab_entry.update(bit_stats.analyse_data())
                tab.append(tab_entry)
                codes_release(handle)
            f.close()
        else:
            print(f"File extension {ext} is not supported")

    # print(tab)
    df = pd.DataFrame(tab)
    print(df)

    df.to_csv("output.csv", index=False)
