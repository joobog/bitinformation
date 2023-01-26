#!/usr/bin/env python3

import argparse
import sys
import struct
import pathlib
import os

from table import *
from parameters import *
from tool import *


def get_preprocessor_from_cache(cache, name, **kwargs):
    ''' Cache preprocessors if not in cache'''
    if name == 'raw':
        key = name
        if key not in cache:
            cache[key] = RawPreprocessor()
    elif name == 'simple':
        key = f"{name}_{kwargs['simple_nbits']}_{kwargs['simple_nvalues']}"
        if key not in cache:
            cache[key] = SimplePackingPreprocessor(nbits=kwargs['simple_nbits'], nelems=kwargs['simple_nvalues'])
    elif name == 'scale':
        key = f"{name}_{kwargs['scale_nbits']}"
        if key not in cache:
            cache[key] = ScalePreprocessor(nbits=kwargs['scale_nbits'])
    else:
        raise Exception(f'Unsupported preprocessor {name}')
    return cache[key]


def get_analyser_from_cache(cache, name, **kwargs):
    if name == 'mask':
        mask = (~data.dtype.type(0x0)) << np.uint8(kwargs['mask_nbits'])
        analyser = MaskAnalyser(data[:size], mask)
    elif name == 'bitinfo':
        analyser = BitInformationAnalyser(szi=kwargs['bitinfo_szi'], confidence=kwargs['bitinfo_confidence'], precision=kwargs['bitinfo_precision'])
    else:
        raise Exception(f'Unsupported analyser {name}')
    return analyser


def get_iteration_steps(self, handle, start, stop, step):
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General
    # parser.add_argument('pos', nargs='+', help='command args ...', type=str)
    parser.add_argument('--stats', help='GRIB file list', nargs='+', type=str)
    parser.add_argument('--compare', help='GRIB files', nargs=2, type=str)
    parser.add_argument('-v', '--verbosity', help='Verbosity level [0|1|2]', type=int, default=1)
    parser.add_argument('-o', '--output', help='Output file name. Extends existing file.', type=str)
    parser.add_argument('-f', '--csv-format', help='Export parameters to csv format (wide | long)', default='long', type=str)
    parser.add_argument('-i', '--iter-step', help='TODO', type=int)
    parser.add_argument('--params', help='Parameters file name', default='parameters.yaml', type=str)
    parser.add_argument('--mode', help='rdonly|update', default='rdonly', type=str)
    # Split keys
    parser.add_argument('--primary-key', help='Primary key consists of multiple keys', nargs='+', default=['short_name', 'stream', 'precision'], type=str)
    # Preprocessor
    parser.add_argument('--preprocessor', help='Preprocessor: raw | simple | scale', default='simple', type=str)
    parser.add_argument('--simple-nbits', help='Number of bits [1..64]', type=int)
    parser.add_argument('--simple-nvalues', help='Use the first n values from the dataset', type=int)
    parser.add_argument('--scale-nbits', help='Number of bits [1..64]', default=16, type=int)
    # Analyser
    parser.add_argument('--analyser', help='Analyser: manual | bitinfo', default='bitinfo', type=str)
    parser.add_argument('--bitinfo-precision', help='Precision', type=float, default=1.0)
    parser.add_argument('--bitinfo-confidence', help='Confidence', type=float, default=0.99)
    parser.add_argument('--bitinfo-szi', help='Disable set zero insignificant', action="store_true")
    parser.add_argument('--mask-nbits', help='Number of the least significant bits set to zero', type=int)

    args = parser.parse_args()

    # Init

    if args.verbosity >= 2:
        print('Configuration')
        print(args)

    if args.output is not None:
        if os.path.exists(args.output):
            output = pd.read_csv(args.output)
            print(output.head)
        else:
            output = pd.DataFrame()

    if args.csv_format == 'wide':
        tab = WideTable()
    elif args.csv_format == 'long':
        tab = LongTable()
    else:
        print(f'Format {args.csv_format} is not supported')
        sys.exit(1)

    preprocessor_cache = dict()
    analyser_cache = dict()
    params = Parameters(args.params, key_columns=tuple(args.primary_key), value_columns=('nbits',))
    tool = Tool()

    # Statistics
    if args.stats is not None:
        for fn in args.stats:
            print(f'Processing {fn}')
            f = open(fn, 'r')
            while True:
                handle = codes_grib_new_from_file(f)
                if not handle:
                    break
                bits_per_value = codes_get_long(handle, 'bitsPerValue')
                simple_nbits = args.simple_nbits if args.simple_nbits is not None else bits_per_value
                scale_nbits = args.scale_nbits if args.scale_nbits is not None else bits_per_value
                preprocessor = get_preprocessor_from_cache(
                    cache=preprocessor_cache,
                    name=args.preprocessor,
                    simple_nbits=simple_nbits, simple_nvalues=args.simple_nvalues,
                    scale_nbits=scale_nbits
                )
                analyser = get_analyser_from_cache(
                    cache=analyser_cache,
                    name=args.analyser,
                    bitinfo_szi=args.bitinfo_szi, bitinfo_confidence=args.bitinfo_confidence, bitinfo_precision=args.bitinfo_precision,
                    mask_nbits=args.mask_nbits
                )
                metadata, stats = tool.stats(handle, preprocessor, analyser)
                metadata['szi'] = args.bitinfo_szi
                metadata['confidence'] = args.bitinfo_confidence
                metadata['precision'] = args.bitinfo_precision
                tab.add(metadata, stats.bitinformation)
                params.add(metadata, {'nbits':stats.nbits_used})
                codes_release(handle)
            f.close()

        # Updated parameters only if enabled by users
        if args.mode == "rdonly":
            if not os.path.exists(args.params):
                params.save()
        elif args.mode == "update":
            params.save()
        else:
            print(f'Unsupported mode {args.mode}. Exiting.')
            sys.exit(1)

        # Output results to CSV if enabled by user
        if args.output is not None:
            tab.save(args.output)


    # Comparison
    if args.compare is not None:
        if self.__analyser == 'bitinfo':
            analyser = BitInformationAnalyser(szi=self.__szi, confidence=self.__confidence, precision=precision, verbosity=args.verbosity)
        else:
            analyser = MaskAnalyser(masked_lsb=res['nbits'] - res['nbits-used'])

        f1 = open(fn1, 'rb')
        f2 = open(fn2, 'rb')
        while True:
            handle1 = codes_grib_new_from_file(f1)
            handle2 = codes_grib_new_from_file(f2)
            if not handle1 or not handle2:
                if handle1:
                    codes_release(handle1)
                if handle2:
                    codes_release(handle2)
                break
            bits_per_value = codes_get_long(handle, 'bitsPerValue')
            simple_nbits = args.simple_nbits if args.simple_nbits is not None else bits_per_value
            scale_nbits = args.scale_nbits if args.scale_nbits is not None else bits_per_value
            preprocessor = get_preprocessor_from_cache(
                cache=preprocessor_cache,
                name=args.preprocessor,
                simple_nbits=simple_nbits, simple_nvalues=args.simple_nvalues,
                scale_nbits=scale_nbits
            )
            analyser = get_analyser_from_cache(
                cache=analyser_cache,
                name=args.analyser,
                bitinfo_szi=args.bitinfo_szi, bitinfo_confidence=args.bitinfo_confidence, bitinfo_precision=args.bitinfo_precision,
                mask_nbits=args.mask_nbits
            )
            tool.compare(handle1, handle2, preprocessor, analyser)
            codes_release(handle1)
            codes_release(handle2)
        f1.close()
        f2.close()
