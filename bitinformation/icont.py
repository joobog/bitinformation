#!/usr/bin/env python3

import argparse
import sys
import os
import pandas as pd
import numpy as np
from eccodes import *

from table import Table, LongTable, WideTable
from parameters import Parameters
from tool import Tool
from preprocessor import RawPreprocessor, SimplePackingPreprocessor, ScalePreprocessor
from analyser import BitInformationAnalyser, MaskAnalyser
from statistics import Stats
from pandas.errors import IndexingError
from simple_packing import ConstantFieldException


def prepend_label(metadata, label):
    return {label+k: v for k, v in metadata.items()}

    # @staticmethod
    # def collect_metadata(preprocessor_md, analyser_md, info=None):
    #     metadata = dict()
    #     # metadata['timestamp'] = self.__timestamp
    #     # metadata['nbits_total'] = data.itemsize * 8
    #     p_md = {'preprocessor_'+k: v for k, v in preprocessor_md.items()}
    #     metadata.update(p_md)
    #     a_md = {'analyser_'+k: v for k, v in analyser_md.items()}
    #     metadata.update(a_md)

    #     if info is not None:
    #         metadata['nbits_used'] = info['nbits_used'] - (metadata['preprocessor_itemsize'] - metadata['preprocessor_bits_per_value'])
    #     return metadata

def get_metadata(h):
    md = dict()
    md['const'] = codes_get_long(h, 'isConstant')
    md['short_name'] = codes_get_string(h, 'shortName')
    kiter = codes_keys_iterator_new(h, namespace='mars')
    while codes_keys_iterator_next(kiter):
        key = codes_keys_iterator_get_name(kiter)
        value = codes_get_string(h, key)
        md[key] = value
    codes_keys_iterator_delete(kiter)
    return md

def get_preprocessor_from_cache(cache, name, **kwargs):
    ''' Cache preprocessors if not in cache'''
    if name == 'raw':
        key = name
        if key not in cache:
            cache[key] = RawPreprocessor(bits_per_value=kwargs['bits_per_value'], nelems=kwargs['nvalues'])
    elif name == 'simple':
        key = f"{name}_{kwargs['bits_per_value']}_{kwargs['nvalues']}"
        if key not in cache:
            cache[key] = SimplePackingPreprocessor(bits_per_value=kwargs['bits_per_value'], nelems=kwargs['nvalues'])
    elif name == 'scale':
        key = f"{name}_{kwargs['bits_per_value']}"
        if key not in cache:
            cache[key] = ScalePreprocessor(bits_per_value=kwargs['bits_per_value'], nelems=kwargs['nvalues'])
    else:
        raise Exception(f'Unsupported preprocessor {name}')
    return cache[key]


def get_analyser_from_cache(cache, name, **kwargs):
    if name == 'mask':
        # mask = (~data.dtype.type(0x0)) << np.uint8(kwargs['mask_nbits'])
        key = f"{name}_{kwargs['mask']}"
        if key not in cache:
            cache[key] = MaskAnalyser(mask=int(kwargs['mask'], 16))
    elif name == 'bitinfo':
        key = f"{name}_{kwargs['bitinfo_szi']}_{kwargs['bitinfo_confidence']}_{kwargs['bitinfo_precision']}"
        if key not in cache:
            cache[key] = BitInformationAnalyser(
                szi=kwargs['bitinfo_szi'],
                confidence=kwargs['bitinfo_confidence'],
                precision=kwargs['bitinfo_precision']
            )
    else:
        raise Exception(f'Unsupported analyser {name}')
    return cache[key]


# def get_iteration_steps(self, handle, start, stop, step):
#     sizes = np.empty(0, dtype=np.uint64)
#     nvalues_total = np.uint64(codes_get_long(handle, 'numberOfValues'))
#     if args.iter_step is not None:
#         start = args.iter_step
#         stop = nvalues_total
#         step = args.iter_step

#         sizes = np.arange(start=start, stop=stop, step=step, dtype=np.uint64)
#         if stop == nvalues_total:
#             sizes = np.append(sizes, nvalues_total)
#     else:
#         sizes = np.append(sizes, nvalues_total)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # General
    # parser.add_argument('pos', nargs='+', help='command args ...', type=str)
    parser.add_argument('--stats', help='GRIB file list', nargs='+', type=str)
    parser.add_argument('--compare', help='GRIB files', nargs=2, type=str)
    # parser.add_argument('--show-parameters', help='GRIB file list', nargs='+', type=str) # TODO
    parser.add_argument('-v', '--verbosity', help='Verbosity level [0|1|2]', type=int, default=1)
    parser.add_argument('--csv', help='CSV output file. (This option appends rows to existing files.)', type=str)
    parser.add_argument('--output', help='Grib file using optimal number of bits.', type=str)
    parser.add_argument('-f', '--csv-format', help='Export parameters to csv format (wide | long)', default='long', type=str)
    # parser.add_argument('-i', '--iter-step', help='TODO', type=int) # TODO
    parser.add_argument('--fallback-to-analyser', help='If enabled, analyser will be used for comparison, if parameter is not available.', action="store_true")
    parser.add_argument('--use-analyser', help='If enabled, analyser will be used for comparison.', action="store_true")

    # Parameters
    parser.add_argument('--parameters', help='Parameters file name', default='parameters.yaml', type=str)
    parser.add_argument('--update-existing-parameters', help='Update parameters.', action="store_true")
    parser.add_argument('--add-missing-parameters', help='Add missing parameters parameters.', action="store_true")
    parser.add_argument('--primary-keys', help='Primary key consists of multiple keys', nargs='+', default=['short_name', 'stream', 'analyser_precision'], type=str)
    parser.add_argument('--value-keys', help='Only one key is supported at the moment', nargs='+', default=['nbits_used', 'preprocessor_bits_per_value'], type=str)

    # Preprocessor
    parser.add_argument('--preprocessor', help='Preprocessor: raw | simple | scale', default='simple', type=str)
    parser.add_argument('-n', '--nvalues', help='Number of bits [1..64]', type=int)
    parser.add_argument('--nbits', help='Number of bits [1..64]', type=int)

    # Analyser
    parser.add_argument('--analyser', help='Analyser: mask | bitinfo', default='bitinfo', type=str)
    parser.add_argument('--bitinfo-precision', help='Precision', type=float, default=1.0)
    parser.add_argument('--bitinfo-confidence', help='Confidence', type=float, default=1.0)
    parser.add_argument('--bitinfo-szi', help='Disable set zero insignificant', action="store_true")
    parser.add_argument('--mask', help='Number of the least significant bits set to zero', type=str)

    args = parser.parse_args()

    # Uncomment this code block to enable updating and adding new parameters by default:
    # if (args.use_analyser or args.fallback_to_analyser) and (not args.add_missing_parameters and not args.update_existing_parameters):
    #     args.add_missing_parameters = True
    #     args.update_existing_parameters = True

    # Init
    if args.verbosity >= 2:
        print('Configuration')
        print(args)

    if args.csv is not None:
        if os.path.exists(args.csv):
            csv = pd.read_csv(args.csv)
            # print(csv.head)
        else:
            csv = pd.DataFrame()


    preprocessor_cache = dict()
    analyser_cache = dict()
    parameters = Parameters(args.parameters, key_columns=tuple(args.primary_keys), value_columns=tuple(args.value_keys))
    tool = Tool()

    # Statistics
    if args.stats is not None:
        if args.csv_format == 'wide':
            tab = WideTable()
        elif args.csv_format == 'long':
            tab = LongTable()
        else:
            print(f'Error: Format {args.csv_format} is not supported')
            sys.exit(1)

        for fn in args.stats:
            if args.verbosity >= 1:
                print(f'Processing {fn}')
            f = open(fn, 'r')
            while True:
                handle = codes_grib_new_from_file(f)
                if not handle:
                    break
                metadata = get_metadata(handle)
                bits_per_value = codes_get_long(handle, 'bitsPerValue')
                if bits_per_value == 0:
                    if args.verbosity >=1:
                        print(f'Skipping constant field')
                else:
                    nbits = args.nbits if args.nbits is not None else bits_per_value
                    preprocessor = get_preprocessor_from_cache(
                        cache=preprocessor_cache,
                        name=args.preprocessor,
                        bits_per_value=nbits,
                        nvalues=args.nvalues
                    )
                    analyser = get_analyser_from_cache(
                        cache=analyser_cache,
                        name=args.analyser,
                        bitinfo_szi=args.bitinfo_szi,
                        bitinfo_confidence=args.bitinfo_confidence,
                        bitinfo_precision=args.bitinfo_precision,
                        mask_nbits=args.nbits
                    )

                    try:
                        info = tool.preprocess_and_analyse(handle, preprocessor, analyser)
                        metadata.update(info)
                        tab.add(metadata)

                        if args.add_missing_parameters:
                            if not parameters.key_exists(metadata):
                                parameters.add(metadata)
                        if args.update_existing_parameters:
                            if parameters.key_exists(metadata):
                                parameters.add(metadata)
                        else:
                            print(f'Error: Unsupported mode {args.mode}. Exiting.')
                            sys.exit(1)
                        parameters.save()

                        if args.output is not None:
                            out_f = open(args.output, 'ab')
                            msg_clone = codes_clone(handle)
                            codes_set_long(msg_clone, "setBitsPerValue", metadata['nbits_used'])
                            codes_write(msg_clone, out_f)
                            out_f.close()
                    except ConstantFieldException:
                        pass
                codes_release(handle)
            f.close()


        # Output results to CSV if enabled by user
        if args.csv is not None:
            tab.save(args.csv)


    # Comparison
    if args.compare is not None:
        tab = Table()
        f1 = open(args.compare[0], 'rb')
        f2 = open(args.compare[1], 'rb')

        while True:
            handle1 = codes_grib_new_from_file(f1)
            handle2 = codes_grib_new_from_file(f2)
            if not handle1 or not handle2:
                if handle1:
                    codes_release(handle1)
                if handle2:
                    codes_release(handle2)
                break
            values1 = codes_get_values(handle1)
            values2 = codes_get_values(handle2)

            if args.nbits is None:
                bits_per_value1 = codes_get_long(handle1, 'bitsPerValue')
                bits_per_value2 = codes_get_long(handle2, 'bitsPerValue')
                assert bits_per_value1 == bits_per_value2
            nbits = args.nbits if args.nbits is not None else bits_per_value1

            preprocessor = get_preprocessor_from_cache(
                cache=preprocessor_cache,
                name=args.preprocessor,
                bits_per_value=nbits,
                nvalues=args.nvalues
            )
            analyser = get_analyser_from_cache(
                cache=analyser_cache,
                name=args.analyser,
                bitinfo_szi=args.bitinfo_szi,
                bitinfo_confidence=args.bitinfo_confidence,
                bitinfo_precision=args.bitinfo_precision,
                mask=args.mask
            )

            md1 = get_metadata(handle1)
            md2 = get_metadata(handle2)
            md1.update(prepend_label(preprocessor.metadata(), 'preprocessor_'))
            md1.update(prepend_label(analyser.metadata(), 'analyser_'))
            md2.update(prepend_label(preprocessor.metadata(), 'preprocessor_'))
            md2.update(prepend_label(analyser.metadata(), 'analyser_'))

            if args.use_analyser: # Force using analysers.
                info1 = tool.preprocess_and_analyse(values1, preprocessor, analyser)
                info2 = tool.preprocess_and_analyse(values2, preprocessor, analyser)
                md1.update(info1)
                md2.update(info2)

                if args.update_existing_parameters or args.add_missing_parameters:
                    if args.update_existing_parameters:
                        if parameters.key_exists(md1):
                            parameters.add(md1)
                        if parameters.key_exists(md2):
                            parameters.add(md2)
                    if args.add_missing_parameters:
                        if not parameters.key_exists(md1):
                            parameters.add(md1)
                        if not parameters.key_exists(md2):
                            parameters.add(md2)

                    parameters.save()
            else: # Force using parameters. If no parameter found, fallback to analysers.
                try:
                    md1.update(parameters.lookup(md1))
                except (KeyError, IndexingError) as e:
                    if args.fallback_to_analyser:
                        # print(f'Warning: No match found in {args.parameters} for {args.compare[1]}. Using analyser \"{analyser.name()}\".')
                        info1 = tool.preprocess_and_analyse(values1, preprocessor, analyser)
                        md1.update(info1)
                        if args.add_missing_parameters:
                            parameters.add(md1)
                            parameters.save()
                    else:
                        raise

                try:
                    md2.update(parameters.lookup(md2))
                except (KeyError, IndexingError) as e:
                    if args.fallback_to_analyser:
                        # print(f'Warning: No match found in {args.parameters} for {args.compare[1]}. Using analyser \"{analyser.name()}\".')
                        info2 = tool.preprocess_and_analyse(values2, preprocessor, analyser)
                        md2.update(info2)
                        if args.add_missing_parameters:
                            parameters.add(md2)
                            parameters.save()
                    else:
                        raise


            comparison = tool.compare(values1, values2, md1, md2, preprocessor)

            m1_stats = get_metadata(handle1)
            m1_stats['filename'] = args.compare[0]
            m1_stats['min'] = values1.min()
            m1_stats['max'] = values1.max()
            m1_stats['mean'] = values1.mean()
            m1_stats['median'] = np.median(values1)

            m2_stats = get_metadata(handle2)
            m2_stats['filename'] = args.compare[1]
            m2_stats['min'] = values2.min()
            m2_stats['max'] = values2.max()
            m2_stats['mean'] = values2.mean()
            m2_stats['median'] = np.median(values2)

            stats = dict()
            stats.update(prepend_label(m1_stats, 'm1_'))
            stats.update(prepend_label(m2_stats, 'm2_'))
            stats.update(prepend_label(tool.compute_stats(values1, values2), 'orig_'))
            # stats.update(prepend_label(tool.compute_stats(comparison['data1'], comparison['data2']), 'masked_'))

            tab.add(stats)
            codes_release(handle1)
            codes_release(handle2)


        # Output results to CSV if enabled by user
        if args.csv is not None:
            tab.save(args.csv)
        f1.close()
        f2.close()

        assert comparison['status'] is not None
        sys.exit(not comparison['status'])
