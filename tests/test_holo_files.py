#!python

import os
import subprocess
import time
import difflib
import pytest
import fnmatch
import json
from typing import List, Tuple
import logging

from .constant_name import *
from . import holo

DEEP_COMPARE = True

HOLOVIBES_BIN = os.path.join(
    os.getcwd(), "build/bin/Holovibes.exe")

assert os.path.isfile(
    HOLOVIBES_BIN), "Cannot find Holovibes.exe, Change the HOLOVIBES_BIN var"


# Create a named logger
logger = logging.getLogger("test_holo")
logger.setLevel(logging.INFO)

# Create a console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

# Set the formatter for the console handler
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
datefmt='%m/%d/%Y %I:%M:%S%p')
console_handler.setFormatter(formatter)

# Add the console handler to the logger
logger.addHandler(console_handler)


def read_holo(path: str) -> holo.HoloFile:
    return holo.HoloFile.from_file(path)

def read_time(path: str) -> float:
    with open(path, "r") as f:
        return float(f.readline())

def write_time(time: float, path: str) -> None:
    with open(path, "w") as file:
        file.write(str(time))


def read_holo_lazy(path: str) -> Tuple[bytes, bytes, bytes]:
    holo_file = holo.HoloLazyReader(path)
    data = holo_file.get_all_bytes()
    holo_file.close()
    return data


def get_cli_arguments(cli_argument_path: str) -> List[str]:
    if not os.path.isfile(cli_argument_path):
        return []

    with open(cli_argument_path, "rb") as f:
        return json.load(f)


def generate_holo_from(folder: str, input: str, output: str, output_error: str, cli_argument: str, config: str = None) -> time.time:
    t1 = time.time()

    # Run holovibes on file
    cmd = [HOLOVIBES_BIN, "-i", input, "-o", output] + get_cli_arguments(cli_argument)
    if config:
        cmd += ['--compute_settings', config]

    logger.info(f"\n Running: {' '.join(cmd)}")

    sub = subprocess.run(cmd, stderr=subprocess.PIPE)

    if sub.returncode != 0:
        with open(output_error, "w") as f_out:
            f_out.write(f"{sub.returncode}\n{sub.stderr.decode('utf-8')}")
    
    os.makedirs("test_logs", exist_ok=True)
    with open(os.path.join("test_logs", "all_errcode.txt"), "a") as f_all:
        f_all.write(f"=== {folder} ===\nReturn: {sub.returncode}\n{sub.stderr.decode('utf-8')}\n")

    t2 = time.time()
    return (t2 - t1)

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

def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    return [n for n in fnmatch.filter(os.listdir(base), pattern) if
        os.path.isfile(os.path.join(base, n))]

def test_holo(folder: str):

    print(folder)

    path = os.path.join(TESTS_DATA, folder)
    input = os.path.join(path, INPUT_FILENAME)
    output_error = os.path.join(path, OUTPUT_ERROR_FILENAME)
    output = os.path.join(path, OUTPUT_FILENAME)
    ref_error = os.path.join(path, ERROR_FILENAME)
    ref = os.path.join(path, REF_FILENAME)
    cli_argument = os.path.join(path, CLI_ARGUMENT_FILENAME)
    config = os.path.join(path, CONFIG_FILENAME)

    error_wanted = False

    def not_found(filename: str) -> None:
        pytest.skip(
            f"Did not find the {filename} file in folder {path}")

    if not os.path.isfile(input):
        input = get_input_file(path)
        if input is None:
            not_found(INPUT_FILENAME)

    if os.path.isfile(ref_error):
        error_wanted = True
    elif find_files(path, "[0-9]*_" + REF_FILENAME) == []:
        not_found(REF_FILENAME)

    if not os.path.isfile(config):
        config = None
        print("Default values might have changed")

    for file in find_files(path, "R_" + OUTPUT_FILENAME):
        os.remove(os.path.join(path, file))

    if os.path.isfile(output_error):
        os.remove(output_error)

    current_time = generate_holo_from(folder, input, output, output_error, cli_argument, config)

    if error_wanted:
        assert os.path.isfile(output_error), f"Should have failed but {OUTPUT_ERROR_FILENAME} not found"
    else:
        assert  find_files(path, "R_" + OUTPUT_FILENAME) != [], f"Should have succeded but {OUTPUT_FILENAME} not found"
        assert not os.path.isfile(output_error), f"Should have succeded but {OUTPUT_ERROR_FILENAME} found"



    if DEEP_COMPARE:
        if error_wanted:
            ref_error_file = open(ref_error, "r")
            ref_error_code = int(ref_error_file.readline())
            ref_error_file.close()

            output_error_file = open(output_error, "r")
            output_error_code = int(output_error_file.readline())
            output_error_file.close()

            assert output_error_code == ref_error_code, f"Return value is invalid: wanted {ref_error_code} but got {output_error_code}"
        else:
            out = read_holo(os.path.join(path,find_files(path, "R_" + OUTPUT_FILENAME)[0]))
            ref = read_holo(os.path.join(path,find_files(path, "[0-9]*_" + REF_FILENAME)[0]))
            try:
                ref_time = read_time(os.path.join(path, "ref_time.txt"))
                logger.info(f"Current time: {current_time} Ref time: {ref_time}")
            except:
                pass

            current_tol, errors = ref.assertHolo(out, path)
            if current_tol != 0.0:
               logger.info(f"Total diff: {current_tol}")

            if len(errors) > 0:
               logger.error(f"Errors: {errors}")
               assert False, f"Errors: {errors}"



    elif not error_wanted: # LAZY_COMPARE
        out = read_holo_lazy(output)
        ref = read_holo_lazy(ref)

        diff_holo(ref, out)
