#!python

import os
import subprocess
import time
import difflib
import pytest
from typing import List, Tuple

import holo

INPUT_FILENAME = "input.holo"
OUTPUT_FILENAME = "last_generated_output.holo"
REF_FILENAME = "ref.holo"
CONFIG_FILENAME = "holovibes.ini"

TESTS_DATA = os.path.join(os.getcwd(), "tests", "data")

HOLOVIBES_BIN = os.path.join(
    os.getcwd(), "build", "Ninja", "Release", "Holovibes.exe")
if not os.path.isfile(HOLOVIBES_BIN):
    HOLOVIBES_BIN = os.path.join(
        os.getcwd(), "build", "Ninja", "Debug", "Holovibes.exe")

assert os.path.isfile(
    HOLOVIBES_BIN), "Cannot find Holovibes.exe, Change the HOLOVIBES_BIN var"


def read_holo(path: str) -> Tuple[bytes, bytes, bytes]:
    holo_file = holo.HoloFileReader(path)
    data = holo_file.get_all()
    holo_file.close()
    return data


def generate_holo_from(input: str, output: str, config: str = None) -> time.time:
    t1 = time.time()

    # Run holovibes on file
    cmd = [HOLOVIBES_BIN, "-i", input, "-o", output]
    if config:
        cmd += ['--ini', config]

    sub = subprocess.run(cmd, stderr=subprocess.PIPE)
    assert sub.returncode == 0, sub.stderr.decode('utf-8')

    t2 = time.time()
    return (t2 - t1),


def diff_holo(a: Tuple[bytes, bytes, bytes], b: Tuple[bytes, bytes, bytes]) -> bool:
    a_header, a_data, a_footer = a
    b_header, b_data, b_footer = b

    # Header
    diffs = list(difflib.diff_bytes(
        difflib.unified_diff, [a_header], [b_header]))
    assert len(diffs) == 0, diffs

    # Footer
    assert a_footer == b_footer, list(
        difflib.ndiff(
            a_footer.decode('ascii').split(","),
            b_footer.decode('ascii').split(",")
        )
    )

    return a != b


def find_tests() -> List[str]:
    return [name for name in os.listdir(TESTS_DATA) if os.path.isdir(os.path.join(TESTS_DATA, name))]

@pytest.mark.parametrize("folder", find_tests())
def test_holo(folder: str):

    path = os.path.join(TESTS_DATA, folder)
    input = os.path.join(path, INPUT_FILENAME)
    output = os.path.join(path, OUTPUT_FILENAME)
    ref = os.path.join(path, REF_FILENAME)
    config = os.path.join(path, CONFIG_FILENAME)

    if not os.path.isfile(input):
        pytest.skip(
            "Did not find the input.holo file in folder {}".format(path))

    if not os.path.isfile(ref):
        pytest.skip(
            "Did not find the ref.holo file in folder {}".format(path))

    if not os.path.isfile(config):
        config = None

    if os.path.isfile(output):
        os.remove(output)

    generate_holo_from(input, output, config)
    out = read_holo(output)
    ref = read_holo(ref)

    diff_holo(out, ref)
