from os.path import basename
from os.path import getsize
from typing import BinaryIO
from struct import pack, unpack
from typing import List

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

class HoloFile:
    def __init__(self, path: str, header: (int, int, int, int)):
        self.width = header[0]
        self.height = header[1]
        self.bytes_per_pixel = header[2]
        self.nb_images = header[3]
        self.path = path

class HoloFileReader(HoloFile):
    def __init__(self, path: str):
        self.io = open(path, 'rb')
        header_bytes = self.io.read(holo_header_size - holo_header_padding_size)
        self.io.read(holo_header_padding_size)

        holo, _version, bits_per_pixel, w, h, img_nb, _data_size, _endianness = unpack(struct_format, header_bytes)
        if holo.decode('ascii') != "HOLO":
            raise Exception('Cannot read holo file')

        header = (w, h, int(bits_per_pixel / 8), img_nb)
        HoloFile.__init__(self, path, header)

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

class HoloFileWriter(HoloFile):
    def __init__(self, path: str, header: (int, int, int, int), data: bytes):
        HoloFile.__init__(self, path, header)
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
        self.io.write(h) # header
        self.io.write(pack(str(holo_header_padding_size) + "s", b'0')) # padding
        self.io.write(self.data) # data
        self.io.write(pack("2s", b'{}')) # empty json footer

    def close(self):
        self.io.close()