#!/usr/bin/env python3

from eccodes import *
import bitinformation as bit
import argparse
import sys

def crop_data(data, szi=True, confidence=0.99, verbose=0):
    bit = bit.BitInformation()
    inf = bit.bitinformation(data, szi=szi, confidence=confidence)

    OT = data.dtype.type

    uintxx = 'uint' + str(data.itemsize*8)
    data_uint = np.frombuffer(data, uintxx)

    if verbose > 0:
        print('Bitinformation')
        print(inf)

    # create a mask for removing the least significant zeros,
    # e.g., 1111.1111.1000.0000
    T = data_uint.dtype.type
    mask = T(0x0)
    for a in reversed(inf):
        if a == 0:
            mask = mask << T(0x1) | T(0x1)
        else:
            break
    mask = ~mask
    if verbose > 0:
        print(binary(mask))

    a = data_uint & mask
    return np.frombuffer(a, OT)

def compare_data(data1, szi=True, confidence=0.99, verbose=0):
    ''' Return 0 if data are equal or 1 if they are not. '''

    data2 = crop_data(data1, szi=szi, confidence=confidence, verbose=verbose)
    if verbose > 0:
        df = pd.DataFrame({'Value1':data1, 'Value2':data2, 'diff':(data1 - data2)}).reset_index()
        df['index'] += 1
        df = df[df['diff'] != 0.0]
        print(df)

    return np.all(data1 == data2)

def compare_grib_files(fn, szi=True, confidence=0.99, verbose=0):
    ''' Return 0 if files are equal, otherwise return 1 '''
    f = open(fn, 'r')

    exit_status = 0
    msg_count = 0;
    while True:
        handle = codes_grib_new_from_file(f)
        if not handle:
            break
        # values = codes_get_values(handle)[0:10]
        values = codes_get_values(handle)
        codes_release(handle)
        msg_status = compare_data(values, szi=szi, confidence=confidence, verbose=verbose)

        if msg_status != 0:
            exit_status = 1
        if verbose > 0:
            msg_count += 1
            msg_status_str = 'GOOD' if msg_status == 1 else 'BAD'
            print(f'Message {msg_count}: {msg_status_str}')
        elif exit_status == 1:
            break
    f.close()
    return exit_status

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gribfile', nargs=1, help='path to a GRIB file', type=str)
    parser.add_argument('-c', '--confidence', help='Confidence', type=float, default=0.99)
    parser.add_argument('-z', '--szi', help='Set zero insignificant', action="store_true")
    parser.add_argument('-v', '--verbose', help='Verbose level', type=int, default=1)
    args = parser.parse_args()
    if args.verbose > 0:
        print(args)

    exit_status = compare_grib_files(args.gribfile[0], szi=args.szi, confidence=args.confidence, verbose=args.verbose)
    sys.exit(exit_status)

