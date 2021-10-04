#!/usr/bin/env python

import sys

import holo

# (number of images, (x_offset, y_offset))
shifts = [
    (10, (0, 0)),
    (10, (10, 10)),
    (10, (20, 10)),
    (10, (50, 10)),
    (10, (120, 30)),
    (10, (120, 30)),
    (10, (80, 80)),
    (10, (0, 0)),
    (10, (-200, -100)),
    (10, (-300, 20)),
    (10, (-400, 200)),
]

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print(f'USAGE: {sys.argv[0]} INPUT OUTPUT')

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    f = holo.HoloLazyReader(input_path)
    data = f.get_frame_by_lines()

    with open(output_path, 'wb') as output_file:
        for shift in shifts:
            nb = shift[0]
            x_off = shift[1][0]
            y_off = shift[1][1]
            shifted_x = []
            shifted_xy = []
            for row in data:
                shifted_x.append(row[x_off:] + row[:x_off])
            shifted_xy = shifted_x[y_off:] + shifted_x[:y_off]
            for _ in range(nb):
                for row in shifted_xy:
                    output_file.write(row)
