import os
import re
import sys

import pytest
from tests.constant_name import find_tests


def pytest_addoption(parser):
    parser.addoption(
        "--specific-tests", action="store", default="", help="List of specific tests to run"
    )

def pytest_configure(config):
    specific_tests = config.getoption("--specific-tests")
    if specific_tests:
        # Split the string into a list. Assuming tests are comma-separated.
        config.specific_tests = specific_tests.split(",")
    else:
        config.specific_tests = []

    # Ensure the directory for log files exists
    log_dir = 'test_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Clear existing logs if they exist to start fresh for each test session
    with open(os.path.join(log_dir, 'all_stdout.txt'), 'w') as f_out, \
         open(os.path.join(log_dir, 'all_stderr.txt'), 'w') as f_err, \
         open(os.path.join(log_dir, 'all_errcode.txt'), 'w') as f_errcode:
        f_out.write('')
        f_err.write('')
        f_errcode.write('')

def pytest_generate_tests(metafunc):
    if 'folder' in metafunc.fixturenames:
        # Assuming the setup from previous steps to parse "--specific-tests"
        specific_tests = getattr(metafunc.config, 'specific_tests', [])

        # Apply the filtered find_tests results to the test function
        metafunc.parametrize("folder", find_tests(specific_tests))



@pytest.fixture(autouse=True)
def capture_output(request, capsys):
    yield  # Let the test run
    captured = capsys.readouterr()
    test_name = request.node.name

    # Append the captured stdout and stderr to the respective files
    log_dir = 'test_logs'
    with open(os.path.join(log_dir, 'all_stdout.txt'), 'a') as f_out, \
         open(os.path.join(log_dir, 'all_stderr.txt'), 'a') as f_err:
        f_out.write(f"=== {test_name} ===\n{captured.out}\n")
        f_err.write(f"=== {test_name} ===\n{captured.err}\n")
