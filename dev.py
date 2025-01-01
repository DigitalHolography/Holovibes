#!/bin/env python

import os
import sys
import subprocess
import fnmatch
import argparse
import subprocess
import webbrowser
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


@goal
def install(args: GoalArgs) -> int:
    build_mode = build_utils.get_build_mode(args.build_mode)
    build_mode = "Debug" if build_mode in ["d", "D", "Debug", "debug"] else "RelWithDebInfo"
    build_dir = build_utils.get_build_dir(args.build_dir)

    # if build dir exist, remove it
    if os.path.isdir(build_dir):
        print(f"Warning: deleting previous build: {build_dir}")
        sys.stdout.flush()
        if subprocess.call(f"rm -rf {build_dir}", shell=True):
            return 1
    if subprocess.call(f"mkdir {build_dir}", shell=True):
        return 1

    cmd = [
        "conan", "install", ".",
        "--build", "missing",
        "--settings", "compiler.cppstd=20",
        "--settings", f"&:build_type={build_mode}",
    ]

    cmd.extend(args.goal_args)

    if args.verbose:
        print("conan cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    try:
        return subprocess.call(cmd)
    except Exception as e:
        print("Please make sure you have installed the build/requirements.txt")
        raise

def conan_build_goal(args: GoalArgs, option: str) -> int:
    build_dir = build_utils.get_build_dir(args.build_dir)

    # Handle prerequisite tasks
    if not os.path.isdir(build_dir):
        sys.stdout.flush()
        if option == "--install" and conan_build_goal(args, "--test"):
            return 1
        if option == "--test" and conan_build_goal(args, "--build"):
            return 1
        if option == "--build" and conan_build_goal(args, "--configure"):
            return 1
        if option == "--configure" and install(args):
            return 1

    # Determine build preset based on mode
    preset = "conan-debug" if args.build_mode in ["d", "D", "Debug", "debug"] else "conan-relwithdebinfo"
    
    cmake_flags = []
    if option == "--test":
        cmake_flags.append("--target test")

    cmake_build_step = []
    if option in ["--build", "--test"]:
        cmake_build_step.append("--build")

    # Construct the full command
    cmd = (
        [os.path.join(DEFAULT_BUILD_FOLDER if args.build_dir is None else args.build_dir, "generators", "conanbuild.bat"), "&&", "cmake"]
        + cmake_build_step
        + ["--preset", preset]
        + cmake_flags
        + args.goal_args
    )

    if args.verbose:
        print("cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    try:
        # Execute the command
        return subprocess.call(cmd)
    except Exception as e:
        print("Did you install build/requirements.txt?")
        raise

@goal
def cmake(args: GoalArgs) -> int:
    return conan_build_goal(args, "--configure")


@goal
def build(args: GoalArgs) -> int:
    return conan_build_goal(args, "--build")


@goal
def test(args: GoalArgs) -> int:
    res = conan_build_goal(args, "--test")

    build_dir = build_utils.get_build_dir(args.build_dir)

    cmd = f"./{build_dir}/test.exe"
    
    try:
        # Execute the command
        return subprocess.call(cmd)
    except Exception as e:
        print("Did you install build/requirements.txt?")
        raise

@goal
def doc(args: GoalArgs) -> int:
    if conan_build_goal(args, "--configure"):
        print("Fail to build project needed for documentation")
        return 1

    build_dir = build_utils.get_build_dir(args.build_dir)

    cmd = ["cmake",
           "--build",
           build_dir,
           "-t",
           "doc"
           ]

    try:
        returnValue = subprocess.call(cmd)
        if not returnValue:
            webbrowser.open(
                "file://" + os.path.realpath("docs/html/index.html"))
        return returnValue
    except:
        print("Failed to build the documentation")
        raise


@goal
def run(args: GoalArgs) -> int:
    if not build_utils.is_windows():
        print("Holovibes is only runnable on Windows")
        return 1

    exe_path = os.path.join(
        build_utils.get_build_dir(args.build_dir),
        RUN_BINARY_FILE,
    )

    # cmd = build_utils.get_conan_venv_start_cmd(args.build_dir, args.generator)
    # cmd.append(exe_path)
    cmd = [exe_path]
    cmd.extend(args.goal_args)

    if args.verbose:
        print("Run cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    out = subprocess.call(cmd)
    return out


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
        print("Please install pytest with '$ python -m pip install pytest'")
        sys.stdout.flush()

    if args.verbose:
        print("Pytest: Running pytest main...")
        sys.stdout.flush()

    return pytest.main(args=["-v", "-o", "log_cli=true"] + args.goal_args)


@goal
def ctest(args: GoalArgs) -> int:

    # cmd = build_utils.get_vcvars_start_cmd(
    #     args.build_env) if build_utils.is_windows() else []

    # build_dir = build_utils.get_build_dir(args.build_dir)
    # previous_path = os.getcwd()

    # os.chdir(build_dir)
    cmd = ["ctest", "--verbose"] + args.goal_args

    if args.verbose:
        print("Ctest cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    out = subprocess.call(cmd)

    os.chdir(previous_path)
    return out


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
    # args.goal_args.extend(["--target", "clean"])
    # conan_build_goal(args, "--build")
    # args.goal_args[:-2]

    # Remove build directory
    if os.path.isdir(os.path.join(DEFAULT_BUILD_FOLDER, DEFAULT_BUILD_BASE)):
        if subprocess.call(f"rm -rf {os.path.join(DEFAULT_BUILD_FOLDER, DEFAULT_BUILD_BASE)}", shell=True):
            return 1

    # Remove last_generated_output.holo from tests/data
    for name in os.listdir(TESTS_DATA):
        path = os.path.join(TESTS_DATA, name)
        last_output_holo = os.path.join(path, OUTPUT_FILENAME)
        last_output_image = os.path.join(path, OUTPUT_FAILED_IMAGE)
        last_ref_image = os.path.join(path, REF_FAILED_IMAGE)

        for file in (last_output_holo, last_output_image, last_ref_image):
            if os.path.isfile(file):
                os.remove(file)

    return 0


@goal
def release(args: GoalArgs) -> int:
    if len(args.goal_args) <= 0:
        print("Please specify part of version to bump (major/minor/patch)")
        return 1

    build_mode = build_utils.get_build_mode(args.build_mode)
    build_dir = build_utils.get_build_dir(args.build_dir)
    bump_part = args.goal_args[0]
    args.build_mode = "Release"
    args.goal_args = []

    if build_utils.bump_all_versions(bump_part):
        return 1

    if os.path.isdir(build_dir):
        print("Build directory found, Running clean goal before release")
        sys.stdout.flush()
        if clean(args):
            return 1

    # run goal conan, cmake, build, test and
    # Get libs paths and add them to the installer file
    conan_build_goal(args, option="--build")

    build_utils.create_release_file(build_dir)

    return 0


@goal
def preRelease(args: GoalArgs) -> int:
    build_mode = build_utils.get_build_mode(args.build_mode)
    build_dir = build_utils.get_build_dir(args.build_dir)
    args.build_mode = "Release"
    args.goal_args = []

    if os.path.isdir(build_dir):
        print("Build directory found, Running clean goal before release")
        sys.stdout.flush()
        if clean(args):
            return 1

    # run goal conan, cmake, build, test and
    # Get libs paths and add them to the installer file
    conan_build_goal(args, option="--build")

    build_utils.create_release_file(build_dir)

    return 0


def run_goal(goal: str, args: GoalArgs) -> int:

    goal_func = GoalsFuncs.get(goal)
    if not goal_func:
        raise Exception(f"Goal {goal} does not exists")

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
        choices=RELEASE_OPT + DEBUG_OPT,
        default="Debug",
        help="Choose between Release mode and Debug mode (Default: Debug)",
    )
    # build.add_argument(
    #     "-g",
    #     choices=NINJA_OPT + NMAKE_OPT + MAKE_OPT,
    #     default=None,
    #     help="Choose between NMake, Make and Ninja (Default: Ninja)",
    # )
    # build.add_argument(
    #     "-t",
    #     choices=CLANG_CL_OPT + CL_OPT,
    #     default=None,
    #     help="Choose between MSVC(CL) and ClangCL (Default: ClangCL)",
    # )

    build_env = parser.add_argument_group("Build environment")
    # build_env.add_argument(
    #     "-p",
    #     help="Path to find the VS Developer Prompt to use to build (Default: auto-find)",
    #     default=None,
    # )
    build_env.add_argument(
        "-i",
        help=f"Path used by cmake to store compiled objects and exe (Default: {DEFAULT_BUILD_BASE}/<generator>/)",
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
