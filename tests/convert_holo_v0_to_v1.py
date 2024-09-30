#!/usr/bin/env python

import json
import struct
import sys
import os.path as path
from os.path import basename, dirname, getsize, splitext

bits_to_bytes = {'8bit': 1, '16bit': 2}

old_struct_format = (
    '='
    '4s'
    'H'                 # unsigned short number of bits per pixels
    'I'                 # unsigned int width of image
    'I'                 # unsigned int height of image
    'I'                 # unsigned int number of images
)

new_struct_format = (
    '='
    '4s'
    'H'                 # unsigned short Version number
    'H'                 # unsigned short number of bits per pixels
    'I'                 # unsigned int width of image
    'I'                 # unsigned int height of image
    'I'                 # unsigned int number of images
    'Q'                 # unsigned long long total data size
    'B'                 # unsigned char endianness
)

old_holo_header_size = 18

padding_size = 35

# Returns (img width, img height, bytes per pixel, number of imgs)
def parse_title(fpath: str) -> Tuple[int, int, int, int, int]:
    print('file is not holo, parsing title information...')
    fname, _ = splitext(basename(fpath))
    elems = fname.split('_')
    w, h, nb, e = elems[(len(elems) - 4):]
    nb = bits_to_bytes[nb]
    e = 0 if e == 'e' else 1
    file_size = getsize(fpath)
    nb_img = file_size / (int(w) * int(h) * nb)
    return (int(w), int(h), nb, int(nb_img), e)

# Returns (img width, img height, bytes per pixel, number of imgs)
def parse_holo(fpath: str) -> Tuple[int, int, int, int, int]:
    print('file is holo, converting from 18 bytes header to 64 bytes')
    with open(fpath, 'rb') as file:
        header = file.read(old_holo_header_size)
        holo, bits_per_pixel, w, h, img_nb = struct.unpack(old_struct_format, header)
        if holo.decode('ascii') != "HOLO":
            raise Exception("Couldn't find HOLO bytes in header")

        bytes_per_pixel = bits_per_pixel // 8

        # skip data
        file.seek(img_nb * w * h * bytes_per_pixel, 1)

        # load metadata
        j = json.load(file)

        return (w, h, bytes_per_pixel, img_nb, j['endianess']) # typo in encoded data

def new_file_path(fpath: str) -> str:
    return path.join(dirname(fpath), f'new_{basename(fpath)}')


if len(sys.argv) != 2:
    exit(1)

fpath = sys.argv[1]

is_holo = True
nfpath = new_file_path(fpath) # don't overwrite file
path_no_ext, ext = splitext(nfpath)
if ext != '.holo':
    is_holo = False
    nfpath = path_no_ext + '.holo' # switch to holo extension if it was a raw file


width, height, bytes_per_pixel, img_nb, endianness = parse_holo(fpath) if is_holo else parse_title(fpath)

with open(nfpath, 'wb') as new_file, open(fpath, 'rb') as file:
    # write new header
    header = struct.pack(
        new_struct_format + f'{padding_size}s', # padding to reserve 64 bytes total for the header
        b'HOLO',
        1,
        bytes_per_pixel * 8, # bits per pixel
        width,
        height,
        img_nb,
        img_nb * width * height * bytes_per_pixel,
        endianness,
        bytes([0] * padding_size)
    )
    new_file.write(header)

    # skip old header before copying data
    if is_holo:
        file.read(18)

    file_size = getsize(fpath)
    total = 18 if is_holo else 0
    while True:
        data = file.read(2**21)
        if data:
            total += new_file.write(data)
            print(f'\r{total / file_size:.2%}', end='', flush=True)
        else:
            print()
            break

print(f'DONE ! Converted {fpath} to {nfpath}')
