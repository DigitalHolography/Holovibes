#!/usr/bin/env python

"""
From the first image of a provided file, this builds a file containing
10 times the first image, then 10 times that image, shifted by SHIFT pixels.
"""

import sys

import holo

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print(f'USAGE: {sys.argv[0]} SHIFT INPUT OUTPUT')

    shift = int(sys.argv[1])
    input_path = sys.argv[2]
    output_path = sys.argv[3]

    f = holo.HoloLazyReader(input_path)

    data = f.get_frame_by_lines()
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
