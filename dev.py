#!/bin/env python

import os
import sys
import subprocess
import fnmatch
import argparse
from dataclasses import dataclass
from typing import List

from tests.constant_name import *
from build.build_constants import *
from build import build_utils

DEFAULT_GOAL = "build"


@dataclass
class GoalArgs:
    build_mode: str
    generator: str
    toolchain: str
    build_env: str
    build_dir: str
    verbose: bool
    goal_args: List[str]


GoalsFuncs = {}


def goal(func, name: str = None):
    GoalsFuncs[name or func.__name__] = func
    return func


# ----------------------------------#
# Goals                            #
# ----------------------------------#


def invoke_preset(args: GoalArgs, action: str, preset: str = None) -> int:
    if args.build_dir is not None:
        raise ValueError("-i is replaced by CMake presets; see docs/DEVELOPMENT.md")
    forwarded = args.goal_args[1:] if args.goal_args[:1] == ["--"] else args.goal_args
    cmd = [
        "powershell.exe", "-NoLogo", "-NoProfile", "-ExecutionPolicy", "Bypass",
        "-File", os.path.join(os.path.dirname(os.path.abspath(__file__)), "dev.ps1"),
        "-Action", action, "-Preset", preset or build_utils.get_preset(args.build_mode),
    ] + (["--"] + forwarded if forwarded else [])
    if args.verbose:
        print(subprocess.list2cmdline(cmd), flush=True)
    return subprocess.call(cmd)


@goal
def install(args: GoalArgs) -> int:
    return invoke_preset(args, "configure")


@goal
def cmake(args: GoalArgs) -> int:
    return invoke_preset(args, "configure")


@goal
def build(args: GoalArgs) -> int:
    return invoke_preset(args, "build")


@goal
def test(args: GoalArgs) -> int:
    return invoke_preset(args, "test", "windows-tests")


@goal
def run(args: GoalArgs) -> int:
    return invoke_preset(args, "run")


@goal
def pytest(args: GoalArgs) -> int:
    directory = "tests/data"

    # delete old output files
    for name in os.listdir(directory):
        path = os.path.join(directory, name)
        last_output_holo = os.path.join(path, OUTPUT_FILENAME)
        last_output_image = os.path.join(path, OUTPUT_FAILED_IMAGE)
        last_ref_image = os.path.join(path, REF_FAILED_IMAGE)
        last_diff_image = os.path.join(path, DIFF_FAILED_IMAGE)
        last_error= os.path.join(path, OUTPUT_ERROR_FILENAME)

        for file in (last_output_holo, last_output_image, last_ref_image, last_diff_image, last_error):
            if os.path.isfile(file):
                os.remove(file)
    try:
        import pytest
    except ImportError as e:
        print(e)
        print("Please install requirements.txt in a Python virtual environment")
        return 1

    if args.verbose:
        print("Pytest: Running pytest main...")
        sys.stdout.flush()

    return pytest.main(args=["-v", "-o", "log_cli=true"] + args.goal_args)


@goal
def ctest(args: GoalArgs) -> int:
    return test(args)



def find_files(base, pattern):
    '''Return list of files matching pattern in base folder.'''
    return [n for n in fnmatch.filter(os.listdir(base), pattern) if
        os.path.isfile(os.path.join(base, n))]

@goal
def build_ref(args: GoalArgs) -> int:
    from tests.test_holo_files import generate_holo_from, write_time

    for name in args.goal_args or find_tests():
        path = os.path.join(TESTS_DATA, name)
        if not os.path.isdir(path):
            print(f"Did not find test dir named: {path}")
            continue

        input = os.path.join(path, INPUT_FILENAME)
        ref_error = os.path.join(path, ERROR_FILENAME)
        ref = os.path.join(path, REF_FILENAME)
        ref_time_path = os.path.join(path, REF_TIME_FILENAME)
        cli_argument = os.path.join(path, CLI_ARGUMENT_FILENAME)
        config = os.path.join(path, CONFIG_FILENAME)

        if not os.path.isfile(input):
            input = get_input_file(path)
            if input is None:
                print(
                    f"Did not find the {INPUT_FILENAME} file in folder {path}")

        if not os.path.isfile(config):
            config = None

        for file in find_files(path, "R_" + REF_FILENAME):
            os.remove(os.path.join(path, file))

        if os.path.isfile(ref_error):
            os.remove(ref_error)

        print(name)
        ref_time = generate_holo_from(path, input, ref, ref_error, cli_argument, config)
        write_time(ref_time, ref_time_path)

    return 0


@goal
def clean(args: GoalArgs) -> int:
    return invoke_preset(args, "clean")


@goal
def release(args: GoalArgs) -> int:
    if len(args.goal_args) != 1 or args.goal_args[0] not in ("major", "minor", "patch"):
        print("Specify a version component: major, minor, or patch")
        return 1
    if build_utils.bump_all_versions(args.goal_args[0]):
        return 1
    args.goal_args = []
    return invoke_preset(args, "package", "windows-release")


@goal
def preRelease(args: GoalArgs) -> int:
    return invoke_preset(args, "package", "windows-release")



def run_goal(goal: str, args: GoalArgs) -> int:

    goal_func = GoalsFuncs.get(goal)
    if not goal_func:
        raise Exception(f"Goal {goal} does not exists")

    os.environ["HOLOVIBES_BIN"] = os.path.abspath(os.path.join(
        build_utils.get_build_dir(args.build_dir, args.build_mode), RUN_BINARY_FILE))
    out = goal_func(args)
    if out != 0:
        print(f"Goal {goal} Failed (out: {out})")
        sys.stdout.flush()
        exit(out)


# ----------------------------------#
# CLI                               #
# ----------------------------------#


def parse_args():
    parser = argparse.ArgumentParser(
        description="Holovibes Dev Tool (only runnable from project root)"
    )

    build = parser.add_argument_group("Build Arguments")
    build.add_argument(
        "-b",
        choices=RELEASE_OPT + DEBUG_OPT + DEV_OPT,
        default="dev",
        help="Choose dev (RelWithDebInfo), Debug, or Release (default: dev)",
    )
    build_env = parser.add_argument_group("Build environment")
    build_env.add_argument(
        "-i",
        help="Deprecated: use CMake presets to choose a build directory",
        default=None,
    )

    parser.add_argument("-v", action="store_true",
                        help="Activate verbose mode")

    args, leftovers = parser.parse_known_args()

    all_goals = list(GoalsFuncs.keys())
    goals = {}

    if len(leftovers) == 0:
        return args, {DEFAULT_GOAL: []}

    current_goal = DEFAULT_GOAL
    for arg in leftovers:
        if arg in all_goals:
            current_goal = arg
            goals[current_goal] = []
        elif current_goal in goals:
            goals[current_goal].append(arg)
        else:
            raise Exception(f"Goal {arg} does not exist")

    return args, goals


if __name__ == "__main__":
    args, goals = parse_args()

    for goal, goal_args in goals.items():
        run_goal(
            goal, GoalArgs(args.b, None, None, None,
                           args.i, args.v, goal_args)
        )

    exit(0)
