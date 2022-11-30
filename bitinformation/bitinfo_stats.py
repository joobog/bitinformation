#!/usr/bin/env python3

from preprocessor import *
from eccodes import *
from bitinformation import *
from statistics import *
import numpy as np
import pandas as pd
import argparse
import sys
import struct
import pathlib
import os
import time
from datetime import datetime



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General
    parser.add_argument('pos', nargs='+', help='command args ...', type=str)
    parser.add_argument('-q', '--quiet', help='Disable output', action="store_true")
    parser.add_argument('-v', '--verbosity', help='Verbosity level [0|1|2]', type=int, default=1)
    parser.add_argument('-o', '--output', help='Output file name', default='output.csv', type=str)
    parser.add_argument('-f', '--csv-format', help='wide | long', default='long', type=str)
    parser.add_argument('-i', '--iter-step', help='TODO', type=int)
    # Preprocessor
    parser.add_argument('-p', '--preprocessor', help='Preprocessor: raw | simple | scale', default='simple', type=str)
    parser.add_argument('--simple-nbits', help='Number of bits [1..64]', default=16, type=int)
    parser.add_argument('--simple-nvalues', help='Number of values', type=int)
    parser.add_argument('--scale-nbits', help='Number of bits [1..64]', default=16, type=int)
    # Analyser
    parser.add_argument('-a', '--analyser', help='Analyser: manual | bitinfo', default='bitinfo', type=str)
    parser.add_argument('--bitinfo-precision', help='Precision', type=float, default=1.0)
    parser.add_argument('--bitinfo-confidence', help='Confidence', type=float, default=0.99)
    parser.add_argument('--bitinfo-szi', help='Disable set zero insignificant', action="store_true")
    parser.add_argument('--cutlsb-nbits', help='Number of the least significant bits set to zero ', type=int)

    args = parser.parse_args()

    timestamp = datetime.fromtimestamp(time.time()).strftime('%Y%m%d%H%M')

    if args.quiet:
        verbose = args.quiet
    verbose = args.verbosity

    if args.verbosity > 2:
        print(args)

    if os.path.exists(args.output):
        output = pd.read_csv(args.output)
        print(output.head)
    else:
        output = pd.DataFrame()

    if args.preprocessor == 'raw':
        preprocessor = RawPreprocessor()
    elif args.preprocessor == 'simple':
        preprocessor = SimplePackingPreprocessor(nbits=args.simple_nbits, nelems=args.simple_nvalues)
    elif args.preprocessor == 'scale':
        preprocessor = ScalePreprocessor(nbits=args.scale_nbits)
    else:
        raise Exception(f'Unsupported preprocessor {args.preprocessor}')

    tab = list()
    for fn in args.pos:
        print(f'Processing {fn}')
        f = open(fn, 'r')
        while True:
            # ecCodes routines
            handle = codes_grib_new_from_file(f)
            if not handle:
                break
            values = codes_get_values(handle)

            # Preprocess
            try:
                data = preprocessor.convert(values)
            except ConstantFieldException:
                data = np.zeros(values.size, dtype=np.int16)

            # Analyse

            sizes = np.empty(0, dtype=np.uint64)
            nvalues_total = np.uint64(codes_get_long(handle, 'numberOfValues'))
            if args.iter_step is not None:
                start = args.iter_step
                stop = nvalues_total
                step = args.iter_step

                sizes = np.arange(start=start, stop=stop, step=step, dtype=np.uint64)
                if stop == nvalues_total:
                    sizes = np.append(sizes, nvalues_total)
            else:
                sizes = np.append(sizes, nvalues_total)

            for size in np.unique(sizes):
                if args.analyser == 'cutlsb':
                    mask = (~data.dtype.type(0x0)) << np.uint8(args.cutlsb_nbits)
                    # print(f'mask ', bin(mask))
                    analyser = ManualAnalyser(data[:size], mask)
                elif args.analyser == 'bitinfo':
                    analyser = BitInformationAnalyser(data[:size], szi=args.bitinfo_szi, confidence=args.bitinfo_confidence, precision=args.bitinfo_precision)
                    # print(f'mask ', bin(analyser.mask))
                else:
                    raise Exception(f'Unsupported analyser {args.analyser}')

                # handle_clone = codes_clone(handle)
                # codes_set_long(handle_clone, 'bitsPerValue', analyser.nbits_used)
                # optimized_data = codes_get_values(handle_clone)
                # print(optimized_data)

                tab_entry = dict()
                tab_entry['timestamp'] = timestamp
                tab_entry['nbits'] = data.itemsize * 8
                tab_entry['nbits_used'] = analyser.nbits_used
                tab_entry['preprocessor'] = preprocessor.name
                tab_entry['const'] = codes_get_long(handle, 'isConstant')
                tab_entry['short_name'] = codes_get_string(handle, 'shortName')
                tab_entry['nvalues'] = size
                tab_entry['nvalues_total'] = nvalues_total
                tab_entry['szi'] = args.bitinfo_szi
                tab_entry['confidence'] = args.bitinfo_confidence
                # tab_entry['rmse'] = Stats.rmse(data, optimized_data)
                # tab_entry['rrmse'] = Stats.rrmse(data, optimized_data)
                # tab_entry['l_inf'] = Stats.l_inf(data, optimized_data)
                # tab_entry['l_1_norm'] = Stats.l_1_norm(data, optimized_data)
                tab_entry['precision'] = args.bitinfo_precision

                if args.csv_format == 'wide':
                    for idx, value in zip(np.flip(np.arange(0, analyser.bitinformation().size)), analyser.bitinformation()):
                        tab_entry[f'b{idx}'] = value
                    tab.append(tab_entry)
                elif args.csv_format == 'long':
                    for idx, value in zip(np.flip(np.arange(0, analyser.bitinformation().size)), analyser.bitinformation()):
                        tab_entry['bitpos'] = idx
                        tab_entry['information'] = value
                        tab.append(tab_entry.copy())
                else:
                    print(f'Format {args.csv_format} is not supported')
                    sys.exit(1)

            codes_release(handle)
        f.close()

    df = pd.DataFrame(tab)
    res = pd.concat([output, df], axis=0)
    res.to_csv(args.output, index=False)
