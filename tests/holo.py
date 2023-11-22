import json
import os
from os.path import getsize
from struct import pack, unpack
from typing import List, Tuple
from PIL import Image
from PIL import ImageChops
from deepdiff import DeepDiff
import numpy as np

from .constant_name import *

holo_header_version = 3
holo_header_size = 64
holo_header_padding_size = 35

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


class HoloLazyIO:
    def __init__(self, path: str, header: Tuple[int, int, int, int]):
        self.width = header[0]
        self.height = header[1]
        self.bytes_per_pixel = header[2]
        self.nb_images = header[3]
        self.path = path


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

    @staticmethod
    def __get_pillow_raw_mode(endianness: int, bits_per_pixel: int) -> str:
        """ Returns Pillow mode to flush and retrive an image """
        if bits_per_pixel == 24:
            return 'RGB'

        if bits_per_pixel not in (8, 16, 32, 64):
            raise Exception(
                f"Cannot from a good file format to store {bits_per_pixel} bits pixels")

        if endianness == 1:  # Big endian
            return 'F;' + str(bits_per_pixel) + 'B'
        else:               # Little endian
            return 'F;' + str(bits_per_pixel)

    @staticmethod
    def __get_numpy_array(frame: bytes, bits_per_pixel: int, w: int, h: int) -> np.dtype:

        meta_data = {
            8: (np.uint8, (w, h)),
            16: (np.uint16, (w, h)),
            24: (np.uint8, (w, h, 3)),
            32: (np.uint32, (w, h)),
            64: (np.uint64, (w, h)),
        }

        return np.frombuffer(frame, dtype=meta_data[bits_per_pixel][0]).reshape(meta_data[bits_per_pixel][1])

    @classmethod
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
            byte_image = cls.__get_numpy_array(
                io.read(bytes_per_frame), bits_per_pixel, w, h)

            ##image = Image.fromarray(byte_image)
            image = byte_image
            
            data.add_frame(image)

        footer_bytes = io.read(
            getsize(path) - holo_header_size - bytes_per_frame * img_nb)
        if len(footer_bytes) != 0:
            data.fill_footer(json.loads(footer_bytes))

        io.close()

        return data

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
            io.write(json.dumps(self.footer).encode('utf-8'))  # Footer

        io.close()

    def assert_footer(ref, chal: "HoloFile"):
        ddiff = DeepDiff(ref.footer, chal.footer,
                         ignore_order=True,
                         significant_digits=5,
                         exclude_paths=["root['info']['input_fps']", ]
                         )

        if 'values_changed' in ddiff:
            if "root['compute settings']['view']['window']['xy']['contrast']['max']" in ddiff['values_changed']:
                diff = ddiff["values_changed"]["root['compute_settings']['view']['window']['xy']['contrast']['max']"]
                if abs(diff["new_value"] / diff["old_value"] - 1) <= CONTRAST_MAX_PERCENT_DIFF:
                    del ddiff["values_changed"]["root['compute_settings']['view']['window']['xy']['contrast']['max']"]
            if "root['compute settings']['view']['window']['xy']['contrast']['min']" in ddiff['values_changed']:
                diff = ddiff["values_changed"]["root['compute_settings']['view']['window']['xy']['contrast']['min']"]
                if abs(diff["new_value"] / diff["old_value"] - 1) <= CONTRAST_MAX_PERCENT_DIFF:
                    del ddiff["values_changed"]["root['compute_settings']['view']['window']['xy']['contrast']['min']"]

            if not ddiff['values_changed']:
                del ddiff['values_changed']

        assert not ddiff, ddiff

    def assertHolo(ref, chal: "HoloFile", basepath: str):

        def __assert(lhs, rhs, name: str):
            assert lhs == rhs, f"{name} differ: {lhs} != {rhs}"

        for attr in ('width', 'height', 'bytes_per_pixel', 'nb_images'):
            __assert(getattr(ref, attr), getattr(chal, attr), attr)

        # ref.assert_footer(chal)

        for i, (l_image, r_image) in enumerate(zip(ref.images, chal.images)):
            diffMatrix = (np.array(l_image) == np.array(r_image))
            diff = np.any(diffMatrix == False)
            #TODO print diff matrix
            #if diff:
            #    l_image.save(os.path.join(basepath, REF_FAILED_IMAGE))
            #    r_image.save(os.path.join(basepath, OUTPUT_FAILED_IMAGE))

            assert not diff


class HoloLazyReader(HoloLazyIO):
    def __init__(self, path: str):
        self.path = path
        self.io = open(path, 'rb')
        header_bytes = self.io.read(
            holo_header_size - holo_header_padding_size)
        self.io.read(holo_header_padding_size)

        holo, _version, bits_per_pixel, w, h, img_nb, _data_size, _endianness = unpack(
            struct_format, header_bytes)
        # if holo.decode('ascii') != "HOLO":
        # self.io.close()
        #raise Exception('Cannot read holo file')

        header = (w, h, int(bits_per_pixel / 8), img_nb)
        HoloLazyIO.__init__(self, path, header)

    def get_all_bytes(self) -> Tuple[bytes, bytes, bytes]:
        data_total_size = self.nb_images * self.height * self.width * self.bytes_per_pixel
        self.io.seek(0)
        h = self.io.read(holo_header_size)
        c = self.io.read(data_total_size)
        f = self.io.read(getsize(self.path) -
                         holo_header_size - data_total_size)
        return h, c, f

    def get_all_frames(self) -> bytes:
        data_total_size = self.nb_images * self.height * self.width * self.bytes_per_pixel
        return self.io.read(data_total_size)

    def get_frame(self) -> List[int]:
        data = []
        for _ in range(self.height * self.width):
            pixel = self.io.read(self.bytes_per_pixel)
            pixel_int = int.from_bytes(pixel, byteorder='big', signed=False)
            data.append(pixel_int)
        return data

    def get_frame_by_lines(self) -> bytes:
        data = []
        for _ in range(self.height):
            data.append(self.io.read(self.bytes_per_pixel * self.width))
        return data

    def close(self):
        self.io.close()


class HoloLazyWriter(HoloLazyIO):
    def __init__(self, path: str, header: Tuple[int, int, int, int], data: bytes):
        HoloLazyIO.__init__(self, path, header)
        self.io = open(path, 'wb')
        self.data = data

    def write(self):
        h = pack(struct_format,
                 b'HOLO',
                 holo_header_version,
                 self.bytes_per_pixel * 8,
                 self.width,
                 self.height,
                 self.nb_images,
                 self.width * self.height * self.nb_images * self.bytes_per_pixel,
                 1)
        self.io.write(h)  # header
        self.io.write(
            pack(str(holo_header_padding_size) + "s", b'0'))  # padding
        self.io.write(self.data)  # data
        self.io.write(pack("2s", b'{}'))  # empty json footer

    def close(self):
        self.io.close()
