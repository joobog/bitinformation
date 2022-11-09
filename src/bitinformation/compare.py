#!/usr/bin/env python3

import bitinformation.bitinformation as bi
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('gribfile', nargs=2, help='path to a GRIB file', type=str)
    parser.add_argument('-c', '--confidence', help='Confidence', type=float)
    args = parser.parse_args()

    return bi.compare_grib_files(fn1, fn2)

