import os
import sys
import json
import jsonpatch
from PIL import Image
import numpy as np
import re

from constant_name import find_tests

from os.path import getsize
from struct import pack, unpack
from typing import List, Tuple

holo_header_version = 5
holo_header_size = 64
holo_header_padding_size = 35
patch_file = 'C:\\Users\\Petrozavodsk\\Desktop\\Stagiaires-2022\\julien.nicolle\\serialized_crash\\Holovibes\\json_patches_holofile\\patch_v4_to_v5.json'
patch_json_file = "C:\\Users\\Petrozavodsk\\Desktop\\Stagiaires-2022\\julien.nicolle\\serialized_crash\\Holovibes\\json_patches_holofile\\patch_update_json_tests.json"

struct_format = (
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

class HoloFile:
    def __init__(self, width: int, height: int, bytes_per_pixel: int) -> None:
        self.width = width
        self.height = height
        self.bytes_per_pixel = bytes_per_pixel
        self.nb_images = 0
        self.images = []
        self.footer = {}


    def add_frame(self, image: Image) -> None:
        self.images.append(image)
        self.nb_images += 1


    def fill_footer(self, kvps: dict) -> None:
        for key, value in kvps.items():
            self.footer[key] = value


    def to_file(self, path: str) -> None:
        io = open(path, 'wb')
        h = pack(struct_format,
                 b'HOLO',
                 holo_header_version,
                 self.bytes_per_pixel * 8,
                 self.width,
                 self.height,
                 self.nb_images,
                 self.width * self.height * self.nb_images * self.bytes_per_pixel,
                 0)
        io.write(h)  # header

        io.write(
            pack(str(holo_header_padding_size) + "s", b'0'))  # padding

        for image in self.images:
            io.write(np.asarray(image).tobytes())

        if len(self.footer.items()) != 0:
            io.write(json.dumps(self.footer, separators=(',', ':')).encode('utf-8'))  # Footer

        io.close()

def __get_numpy_array(frame: bytes, bits_per_pixel: int, w: int, h: int) -> np.dtype:

    meta_data = {
        8: (np.uint8, (w, h)),
        16: (np.uint16, (w, h)),
        24: (np.uint8, (w, h, 3)),
        32: (np.uint32, (w, h)),
        64: (np.uint64, (w, h)),
    }

    return np.frombuffer(frame, dtype=meta_data[bits_per_pixel][0]).reshape(meta_data[bits_per_pixel][1])

def from_file(cls, path: str):
    io = open(path, 'rb')

    # Read header
    header_bytes = io.read(holo_header_size - holo_header_padding_size)
    io.read(holo_header_padding_size)

    # Unpack header data
    holo, _version, bits_per_pixel, w, h, img_nb, _data_size, _endianness = unpack(
        struct_format, header_bytes)

    bytes_per_pixel = bits_per_pixel // 8
    bytes_per_frame = w * h * bytes_per_pixel
    data = cls(w, h, bytes_per_pixel)

    # Add Frames
    for _ in range(img_nb):
        image = Image.fromarray(__get_numpy_array(
            io.read(bytes_per_frame), bits_per_pixel, w, h))
        data.add_frame(image)

    footer_bytes = io.read(
        getsize(path) - holo_header_size - bytes_per_frame * img_nb)
    if len(footer_bytes) != 0:
        data.fill_footer(json.loads(footer_bytes))

    io.close()

    return data

def patch_holofile(path, patch):
    holo = from_file(HoloFile, path)
    try:
        jsonpatch.apply_patch(holo.footer, patch, in_place=True)
    except:
        pass
    holo.to_file(path)


def patch_settings(path, patch):
    with open(path) as f:
        new = jsonpatch.apply_patch(json.loads(f.read()), patch)
    with open(path, "w") as f:
        f.write(json.dumps(new))


if __name__ == "__main__":
    with open(patch_file) as f:
        json_patch_holo = jsonpatch.JsonPatch.from_string(f.read())
    with open(patch_json_file) as f:
        json_patch_json = jsonpatch.JsonPatch.from_string(f.read())

    for f in find_tests():
        path = "./tests/data/" + f
        print(f"converting {f}")
        try:
            patch_holofile(path + "/ref.holo", json_patch_holo)
            patch_settings(path + "/holovibes.json", json_patch_json)
        except:
            print("ok")