from os.path import basename
from os.path import getsize
from typing import BinaryIO
from struct import unpack

bits_to_bytes = {'8bit': 1, '16bit': 2}
holo_header_size = 64

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

padding_size = 35

class FileData:
    def __init__(self, fpath: str, parse_output: (int, int, int, int)):
        self.width = parse_output[0]
        self.height = parse_output[1]
        self.bytes_per_pixel = parse_output[2]
        self.nb_images = parse_output[3]
        self.fpath = fpath
        self.size = getsize(fpath)
        self.is_holo = is_holo(fpath)
        self.io = open(fpath, 'rb')
        if self.is_holo:
            self.io.read(holo_header_size)


    def get_frame(self) -> bytes:
        data = []
        for _ in range(self.height):
            data.append(self.io.read(self.bytes_per_pixel * self.width))
        return data


def is_holo(fpath: str) -> bool:
    ext = fpath.split('.')[-1]
    return ext == "holo"


# Returns (img width, img height, bytes per pixel, number of imgs)
def parse_title(fpath: str) -> (int, int, int, int):
    fname = basename(fpath)
    elems = fname.split('_')
    w, h, nb, _ = elems[(len(elems) - 4):]
    nb = bits_to_bytes[nb]
    file_size = getsize(fpath)
    nb_img = file_size / (int(w) * int(h) * nb)
    return (int(w), int(h), nb, int(nb_img))


# Returns (img width, img height, bytes per pixel, number of imgs)
def parse_holo(fpath: str) -> (int, int, int, int):
    with open(fpath, 'rb') as file:
        header = file.read(holo_header_size)
        holo, _version, bits_per_pixel, w, h, img_nb, _data_size, _endianness = unpack(struct_format, header)
        if holo.decode('ascii') != "HOLO":
            return (0, 0, 0, 0)
        return (w, h, int(bits_per_pixel / 8), img_nb)


def parse_file(fpath: str) -> FileData:
    if is_holo(fpath):
        return FileData(fpath, parse_holo(fpath))
    else:
        return FileData(fpath, parse_title(fpath))
