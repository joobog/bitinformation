#!/usr/bin/env python3

from eccodes import *
import bitinformation as bit
from statistics import *
import numpy as np
import pandas as pd
import argparse
import sys
import struct
import pathlib

class ComparisonTool:
    def __init__(self, szi=True, confidence=0.99, verbose=0):
        self._szi = szi
        self._confidence = confidence
        self._verbose = verbose

    def compare_grib_files(self, fn1, fn2, precision=0.99, n=None):
        ''' Return 0 if files are equal, otherwise return 1 '''
        f1 = open(fn1, 'r')
        f2 = open(fn2, 'r')

        exit_status = 0
        msg_count = 0;
        while True:
            handle1 = codes_grib_new_from_file(f1)
            handle2 = codes_grib_new_from_file(f2)
            if not handle1 or not handle2:
                break
            if n is not None:
                values1 = codes_get_values(handle1)[:n]
                values2 = codes_get_values(handle2)[:n]
            else:
                values1 = codes_get_values(handle1)
                values2 = codes_get_values(handle2)

            codes_release(handle1)
            codes_release(handle2)

            inf1 = bit.BitInformationAnalyser(values1, szi=self._szi, confidence=self._confidence, precision=precision, verbose=self._verbose)
            cleaned_values1 = inf1.cleaned_data()

            inf2 = bit.BitInformationAnalyser(values2, szi=self._szi, confidence=self._confidence, precision=precision, verbose=self._verbose)
            cleaned_values2 = inf2.cleaned_data()

            if self._verbose > 0:
                df = pd.DataFrame({'Value1':values1, 'Value2':values2, 'diff':(cleaned_values1 - cleaned_values2)}).reset_index()
                df['index'] += 1
                df = df[df['diff'] != 0.0]
                if not df.empty:
                    print(df)
                stats = dict()
                stats['RMSE'] = Stats.rmse(values1, values2)
                stats['RRMSE'] = Stats.rrmse(values1, values2)
                stats['L-Inf'] = Stats.l_inf(values1, values2)
                stats['L-1-Norm'] = Stats.l_1_norm(values1, values2)
                stats['ndiff_values'] = df.shape[0]
                print(stats)
            msg_status = np.all(cleaned_values1 == cleaned_values2)

            if msg_status != 0:
                exit_status = 1
            if self._verbose > 0:
                msg_count += 1
                msg_status_str = 'GOOD' if msg_status == 1 else 'BAD'
                print(f'Message {msg_count}: {msg_status_str}')
            elif exit_status == 1:
                break
        f1.close()
        f2.close()
        return exit_status


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gribfilename', nargs=2, help='Grib files ...', type=str)
    parser.add_argument('-n', '--number', help='Test the first n values only', type=int)
    parser.add_argument('-c', '--confidence', help='Confidence', type=float, default=0.99)
    parser.add_argument('-p', '--precision', help='Precision', type=float, default=0.99)
    parser.add_argument('-z', '--szi', help='Disable set zero insignificant', action="store_true")
    parser.add_argument('-v', '--verbosity', help='Verbosity level [0|1|2]', type=int, default=1)
    args = parser.parse_args()
    if args.verbosity >= 1:
        print(args)
    verbose = args.verbosity

    if args.verbosity > 2:
        print(args)

    tool = ComparisonTool(szi=not args.szi, confidence=args.confidence, verbose=verbose)
    exit_status = tool.compare_grib_files(args.gribfilename[0], args.gribfilename[1], n=args.number, precision=args.precision)

    sys.exit(exit_status)

