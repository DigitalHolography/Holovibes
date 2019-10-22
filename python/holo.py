from os.path import basename
from typing import BinaryIO


def load_image(f: BinaryIO, w: int, h: int, nb: int) -> bytes:
    data = []
    for _ in range(h):
        data.append(f.read(nb * w))
    return data


bits_to_bytes = {'8bit': 1, '16bit': 2}


def parse_title(fpath: str) -> (int, int, int):
    fname = basename(fpath)
    elems = fname.split('_')
    w, h, nb, _ = elems[(len(elems) - 4):]
    nb = bits_to_bytes[nb]
    return (int(w), int(h), nb)
