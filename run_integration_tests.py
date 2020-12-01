#!python

import os
import subprocess
import time
import unittest
import difflib

from resources.python import holo

TESTS_DIR = "./tests/"
HOLOVIBES_BIN_DIR = "./build/Ninja/Release/Holovibes.exe"
TMP_GENERATED_HOLO_NAME = "last_generated_output.holo"

def read_holo(file):
    holo_file = holo.HoloFileReader(TESTS_DIR + file)
    data = holo_file.get_all()
    holo_file.close()
    return data

def generate_holo_from(file):
    t1 = time.time()

    # Run holovibes on file
    cmd = [HOLOVIBES_BIN_DIR,
           "-i", TESTS_DIR + file,
           "-o", TESTS_DIR + TMP_GENERATED_HOLO_NAME]
    status = subprocess.call(cmd, stdout=subprocess.DEVNULL)
    if not status == 0:
        raise Exception("Error while running Holovibes to generate test output file.")

    t2 = time.time()
    return (t2 - t1)

def diff_holo(a, b):
    a_header, a_data, a_footer = a
    b_header, b_data, b_footer = b

    # Header
    diffs = difflib.diff_bytes(difflib.unified_diff, [a_header], [b_header])
    for diff in diffs:
        print(diff)

    # Footer
    if not a_footer == b_footer:
        diffs = difflib.ndiff(a_footer.decode('ascii').split(","),
                              b_footer.decode('ascii').split(","))
        for diff in diffs:
            print(diff)

    return not (a == b)

class HoloEqualsAssertions:
    def assertHoloEquals(self, input_path, expected_output_path):
        generate_holo_from(input_path)
        out = read_holo(TMP_GENERATED_HOLO_NAME)
        ref = read_holo(expected_output_path)

        diff = diff_holo(out, ref)
        if not diff == 0:
            raise AssertionError("Output holo is different from ref")
        else:
            # Remove tmp generated .holo only if test passed
            os.remove(TESTS_DIR + TMP_GENERATED_HOLO_NAME)

class TestSimplePCAOnNumbers(unittest.TestCase, HoloEqualsAssertions):

    def test_basic(self):
        self.assertHoloEquals('test_basic/input.holo', 'test_basic/ref.holo')

if __name__ == '__main__':
    unittest.main()