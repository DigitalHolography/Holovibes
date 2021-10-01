#!python

import os
import subprocess
import time
import difflib
import pytest
import json
from typing import List, Tuple

from .constant_name import generate_holo_from, TESTS_DATA, OUTPUT_FILENAME, REF_FILENAME, CONFIG_FILENAME, CLI_ARGUMENT_FILENAME, INPUT_FILENAME, TESTS_DATA
from . import holo

HOLOVIBES_BIN = os.path.join(
    os.getcwd(), "build", "Ninja", "Release", "Holovibes.exe")
if not os.path.isfile(HOLOVIBES_BIN):
    HOLOVIBES_BIN = os.path.join(
        os.getcwd(), "build", "Ninja", "Debug", "Holovibes.exe")

assert os.path.isfile(
    HOLOVIBES_BIN), "Cannot find Holovibes.exe, Change the HOLOVIBES_BIN var"


def read_holo(path: str) -> Tuple[bytes, bytes, bytes]:
    holo_file = holo.HoloLazyReader(path)
    data = holo_file.get_all_bytes()
    holo_file.close()
    return data


def get_cli_arguments(cli_argument_path: str) -> List[str]:
    if not os.path.isfile(cli_argument_path):
        return []

    with open(cli_argument_path, "rb") as f:
        return json.load(f)


def generate_holo_from(input: str, output: str, cli_argument: str, config: str = None) -> time.time:
    t1 = time.time()

    # Run holovibes on file
    cmd = [HOLOVIBES_BIN, "-i", input, "-o", output] + \
        get_cli_arguments(cli_argument)
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

    # Data
    diffs = list(difflib.diff_bytes(
        difflib.unified_diff, [a_data], [b_data]))
    assert len(diffs) == 0, diffs

    # Footer
    assert a_footer == b_footer, list(
        difflib.ndiff(
            a_footer.decode('ascii').split(","),
            b_footer.decode('ascii').split(",")
        )
    )

    return a != b


@pytest.mark.flaky(reruns=5, reruns_delay=3, )
@pytest.mark.parametrize("folder", find_tests())
def test_holo(folder: str):

    path = os.path.join(TESTS_DATA, folder)
    input = os.path.join(path, INPUT_FILENAME)
    output = os.path.join(path, OUTPUT_FILENAME)
    ref = os.path.join(path, REF_FILENAME)
    cli_argument = os.path.join(path, CLI_ARGUMENT_FILENAME)
    config = os.path.join(path, CONFIG_FILENAME)

    def not_found(filename: str) -> None:
        pytest.skip(
            f"Did not find the {filename} file in folder {path}")

    if not os.path.isfile(input):
        input = get_input_file(path)
        if input is None:
            not_found(INPUT_FILENAME)

    if not os.path.isfile(ref):
        not_found(REF_FILENAME)

    if not os.path.isfile(config):
        not_found(CONFIG_FILENAME)

    if os.path.isfile(output):
        os.remove(output)

    generate_holo_from(input, output, cli_argument, config)
    out = read_holo(output)
    ref = read_holo(ref)

    diff_holo(out, ref)
