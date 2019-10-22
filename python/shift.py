#!/usr/bin/env python

"""
From the first image of a provided RAW file, this builds a RAW file containing
10 times the first image, then 10 times that image, shifted by SHIFT pixels.
"""

import sys

import holo

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'USAGE: {sys.argv[0]} SHIFT INPUT.raw OUTPUT.raw')

    shift = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    with open(input_path, 'rb') as input_file:
        data = holo.load_image(input_file, *holo.parse_title(input_path))
        shifted = []
        for row in data:
            shifted.append(row[shift:] + row[:shift])

        with open(output_path, 'wb') as output_file:
            for _ in range(10):
                for row in data:
                    output_file.write(row)
            for _ in range(10):
                for row in shifted:
                    output_file.write(row)
