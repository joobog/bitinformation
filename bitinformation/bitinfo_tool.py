#!/usr/bin/env python3

from eccodes import *
import bitinformation as bit
import numpy as np
import pandas as pd
import argparse
import sys
import struct


class BitInfoTool:
    def __init__(self, szi=True, confidence=0.99, verbose=0):
        self._bitinfo = bit.BitInformation(verbose=verbose)
        self._szi = szi
        self._confidence = confidence
        self._verbose = verbose

    def __binary(self, num):
        return ''.join('{:0>8b}'.format(c) for c in struct.pack('!Q', num))

    def __crop_data(self, data):
        inf = self._bitinfo.bitinformation(data, szi=self._szi, confidence=self._confidence)
        OT = data.dtype.type
        uintxx = 'uint' + str(data.itemsize*8)
        data_uint = np.frombuffer(data, uintxx)

        if self._verbose > 0:
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
        if self._verbose > 0:
            print('Used bit mask: ', self.__binary(mask))

        a = data_uint & mask
        return np.frombuffer(a, OT)


    def __compare_data(self, data1, data2):
        ''' Return 0 if data are equal or 1 if they are not. '''
        data1_cropped = self.__crop_data(data1)
        data2_cropped = self.__crop_data(data2)
        if self._verbose > 0:
            df = pd.DataFrame({'Value1':data1, 'Value2':data2, 'diff':(data1_cropped - data2_cropped)}).reset_index()
            df['index'] += 1
            df = df[df['diff'] != 0.0]
            if not df.empty:
                print(df)

        return np.all(data1_cropped == data2_cropped)

    def __check_data(self, data):
        ''' Return 0 if data are equal or 1 if they are not. '''
        data_cropped = self.__crop_data(data)
        if self._verbose > 0:
            df = pd.DataFrame({'Value1':data, 'Value2':data_cropped, 'diff':(data - data_cropped)}).reset_index()
            df['index'] += 1
            df = df[df['diff'] != 0.0]
            if not df.empty:
                print(df)
        return np.all(data == data_cropped)


    def check_grib_file(self, fn, n=None):
        ''' Return 0 if files are equal, otherwise return 1 '''
        f = open(fn, 'r')

        exit_status = 0
        msg_count = 0;
        while True:
            handle = codes_grib_new_from_file(f)
            if not handle:
                break
            if n is not None:
                values = codes_get_values(handle)[:n]
            else:
                values = codes_get_values(handle)

            codes_release(handle)
            msg_status = self.__check_data(values)

            if msg_status != 0:
                exit_status = 1
            if self._verbose > 0:
                msg_count += 1
                msg_status_str = 'GOOD' if msg_status == 1 else 'BAD'
                print(f'Message {msg_count}: {msg_status_str}')
            elif exit_status == 1:
                break
        f.close()
        return exit_status


    def compare_grib_files(self, fn1, fn2, n=None):
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
                values2 = codes_get_values(handle1)[:n]
            else:
                values1 = codes_get_values(handle1)
                values2 = codes_get_values(handle1)

            codes_release(handle1)
            codes_release(handle2)
            msg_status = self.__compare_data(values1, values2)

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
    parser.add_argument('pos', nargs='+', help='command args ...', type=str)
    parser.add_argument('-n', '--number', help='Test the first n values only', type=int)
    parser.add_argument('-c', '--confidence', help='Confidence', type=float, default=0.99)
    parser.add_argument('-z', '--szi', help='Disable set zero insignificant', action="store_true")
    parser.add_argument('-s', '--silent', help='Disable output', action="store_true")
    parser.add_argument('-v', '--verbosity', help='Verbosity level [0|1|2]', type=int, default=1)
    args = parser.parse_args()
    if args.silent:
        verbose = arg
    verbose = args.verbosity

    if args.verbosity > 2:
        print(args)

    bit = BitInfoTool(szi=not args.szi, confidence=args.confidence, verbose=verbose)
    command = args.pos[0]
    if command == 'check':
        exit_status = bit.check_grib_file(args.pos[1], args.number)
    elif command == 'compare':
        exit_status = bit.compare_grib_files(args.pos[1], args.pos[2], args.number)
    else:
        printf(f'Error: Command {command} is not supported. Exiting')
        sys.exit(1)

    sys.exit(exit_status)

