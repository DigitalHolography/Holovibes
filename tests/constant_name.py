from typing import List
import os.path

INPUT_FILENAME = "input.holo"
OUTPUT_FILENAME = "last_generated_output.holo"
OUTPUT_ERROR_FILENAME = "output_error.txt"
REF_FILENAME = "ref.holo"
REF_TIME_FILENAME = "ref_time.txt"
ERROR_FILENAME = "error.txt"
CONFIG_FILENAME = "holovibes.json"
CLI_ARGUMENT_FILENAME = "cli_argument.json"

OUTPUT_FAILED_IMAGE = "out.png"
REF_FAILED_IMAGE = "ref.png"
DIFF_FAILED_IMAGE = "diff.png"

TESTS_DATA = os.path.join(os.getcwd(), "tests", "data")
TESTS_INPUTS = os.path.join(TESTS_DATA, "inputs")
CONTRAST_MAX_PERCENT_DIFF = 0.05


def find_tests(specific_tests=None) -> List[str]:
    all_tests = [name for name in os.listdir(TESTS_DATA)
                 if os.path.isdir(os.path.join(TESTS_DATA, name)) and name != "inputs"]
    if specific_tests:
        filtered_tests = [name for name in all_tests if name in specific_tests]
        return filtered_tests
    return all_tests

def get_input_file(test_folder: str) -> str:
    general_inputs = os.listdir(TESTS_INPUTS)
    for general_input in general_inputs:
        # Remove ".holo" (5 last char) of the filename
        if general_input[:-5] in test_folder:
            return os.path.join(TESTS_INPUTS, general_input)
    return None
