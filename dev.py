#!/bin/env python

import os
import sys
import subprocess
import argparse
import subprocess
from time import sleep
from collections import namedtuple

from tests.constant_name import *
from build.build_constants import *
from build import build_utils

DEFAULT_GOAL = "build"

GoalArgs = namedtuple(
    "GoalArgs",
    [
        "build_mode",
        "generator",
        "toolchain",
        "build_env",
        "build_dir",
        "verbose",
        "goal_args",
    ],
)
GoalsFuncs = {}


def goal(func, name: str = None):
    GoalsFuncs[name or func.__name__] = func
    return func


# ----------------------------------#
# Goals                            #
# ----------------------------------#


@goal
def conan(args) -> int:
    cmd = []
    generator = build_utils.get_generator(args.generator)
    build_mode = build_utils.get_build_mode(args.build_mode)
    build_dir = build_utils.get_build_dir(args.build_dir, generator)

    # if build dir exist, remove it
    if os.path.isdir(build_dir):
        print(f"Warning: deleting previous build: {build_dir}")
        sys.stdout.flush()
        if subprocess.call(f"rm -rf {build_dir}", shell=True):
            return 1

    cmd += [
        "conan",
        "install",
        ".",
        "-if",
        build_dir,
        "--build",
        "missing",
        "-s",
        f"build_type={build_mode}",
    ] + args.goal_args

    if args.verbose:
        print("conan cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    return subprocess.call(cmd)


@goal
def cmake(args):
    cmd = build_utils.get_vcvars_start_cmd(args.build_env) if build_utils.is_windows() else []
    toolchain = build_utils.get_toolchain(args.toolchain)
    generator = build_utils.get_generator(args.generator)
    build_mode = build_utils.get_build_mode(args.build_mode)
    build_dir = build_utils.get_build_dir(args.build_dir, generator)

    if not os.path.isdir(build_dir):
        print("Build directory not found, Running conan goal before cmake")
        sys.stdout.flush()
        if conan(args):
            return 1

    cmd += [
        "cmake",
        "-B",
        build_dir,
        "-G",
        generator,
        "-S",
        ".",
        "-DCMAKE_VERBOSE_MAKEFILE=OFF",
        f"-DCMAKE_BUILD_TYPE={build_mode}",
        f"-DCMAKE_TOOLCHAIN_FILE={toolchain}",
    ] + args.goal_args

    if args.verbose:
        print("Cmake cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    return subprocess.call(cmd)


@goal
def build(args):
    cmd = build_utils.get_vcvars_start_cmd(args.build_env) if build_utils.is_windows() else []
    build_mode = build_utils.get_build_mode(args.build_mode)
    build_dir = build_utils.get_build_dir(
        args.build_dir, build_utils.get_generator(args.generator)
    )

    if not os.path.isdir(build_dir):
        print("Build directory not found, Running cmake goal before build")
        sys.stdout.flush()
        if cmake(args):
            return 1

    cmd += ["cmake", "--build", build_dir] + args.goal_args

    if args.verbose:
        print("Build cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    return subprocess.call(cmd)


@goal
def run(args):
    build_mode = build_utils.get_build_mode(args.build_mode)
    exe_path = os.path.join(
        build_utils.get_build_dir(
            args.build_dir, build_utils.get_generator(args.generator)
        ),
        build_mode,
        RUN_BINARY_FILE,
    )

    cmd = [
        exe_path,
    ] + args.goal_args

    if args.verbose:
        print("Run cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    out = subprocess.call(cmd)
    return out


@goal
def pytest(args):
    try:
        import pytest
    except ImportError as e:
        print(e)
        print("Please install pytest with '$ python -m pip install pytest'")
        sys.stdout.flush()

    if args.verbose:
        print("Pytest: Running pytest main...")
        sys.stdout.flush()

    return pytest.main(
        args=[
            "-v",
        ]
        + args.goal_args
    )


@goal
def ctest(args):
    cmd = build_utils.get_vcvars_start_cmd(args.build_env) if build_utils.is_windows() else []
    exe_path = args.build_dir or os.path.join(
        DEFAULT_BUILD_BASE, build_utils.get_generator(args.generator), "Holovibes"
    )
    previous_path = os.getcwd()

    os.chdir(exe_path)
    cmd += ["ctest", "--verbose"] + args.goal_args

    if args.verbose:
        print("Ctest cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    out = subprocess.call(cmd)

    os.chdir(previous_path)
    return out


@goal
def build_ref(args) -> int:
    from tests.test_holo_files import generate_holo_from

    dirs = find_tests()
    for name in dirs:
        path = os.path.join(TESTS_DATA, name)
        if (os.path.isdir(path) and len(args.goal_args) == 0) or name in args.goal_args:
            input = os.path.join(path, INPUT_FILENAME)
            ref = os.path.join(path, REF_FILENAME)
            cli_argument = os.path.join(path, CLI_ARGUMENT_FILENAME)
            config = os.path.join(path, CONFIG_FILENAME)

            if not os.path.isfile(input):
                input = get_input_file(path)
                if input is None:
                    print(f"Did not find the {INPUT_FILENAME} file in folder {path}")

            if not os.path.isfile(config):
                config = None

            if os.path.isfile(ref):
                os.remove(ref)

            print(name)
            generate_holo_from(input, ref, cli_argument, config)

    return 0


@goal
def clean(args) -> int:
    # Remove build directory
    if os.path.isdir(DEFAULT_BUILD_BASE):
        if subprocess.call(f"rm -rf {DEFAULT_BUILD_BASE}", shell=True):
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
def release(args) -> int:
    if len(args.goal_args) <= 0:
        print("Please specify part of version to bump")
        return 1

    cmd = []
    generator = build_utils.get_generator(args.generator)
    build_mode = build_utils.get_build_mode(args.build_mode)
    build_dir = build_utils.get_build_dir(args.build_dir, generator)
    bump_part = args.goal_args[0]

    if build_mode != "Release":
        print("Can only create release with a Release build")
        return 1

    if not os.path.isdir(build_dir):
        print("Build directory not found, Running build goal before release")
        sys.stdout.flush()
        if build(args) or ctest(args) or pytest(args):
            return 1

    if not os.path.isdir(INSTALLER_OUTPUT):
        os.mkdir(INSTALLER_OUTPUT)

    cmd += ["conan", "build", ".", "-if", build_dir, "-s", f"build_type={build_mode}"]

    if args.verbose:
        print("conan cmd: {}".format(" ".join(cmd)))
        sys.stdout.flush()

    if subprocess.call(cmd):
        return 1

    paths = build_utils.get_lib_paths()
    if build_utils.bump_all_versions(bump_part):
        return 1

    build_utils.create_release_file(paths)

    return subprocess.call(["iscc", ISCC_FILE])


def run_goal(goal: str, args) -> int:

    goal_func = GoalsFuncs.get(goal)
    if not goal_func:
        raise Exception(f"Goal {goal} does not exists")

    out = goal_func(args)
    if out != 0:
        print(f"Goal {goal} Failed (out: {out})")
        sys.stdout.flush()
        exit(out)


# ----------------------------------#
# CLI                              #
# ----------------------------------#


def parse_args():
    parser = argparse.ArgumentParser(
        description="Holovibes Dev Tool (only runnable from project root)"
    )

    build = parser.add_argument_group("Build Arguments")
    build.add_argument(
        "-b",
        choices=RELEASE_OPT + DEBUG_OPT,
        default=None,
        help="Choose between Release mode and Debug mode (Default: Debug)",
    )
    build.add_argument(
        "-g",
        choices=NINJA_OPT + NMAKE_OPT + MAKE_OPT,
        default=None,
        help="Choose between NMake, Make and Ninja (Default: Ninja)",
    )
    build.add_argument(
        "-t",
        choices=CLANG_CL_OPT + CL_OPT,
        default=None,
        help="Choose between MSVC(CL) and ClangCL (Default: ClangCL)",
    )

    build_env = parser.add_argument_group("Build environment")
    build_env.add_argument(
        "-p",
        help="Path to find the VS Developer Prompt to use to build (Default: auto-find)",
        default=None,
    )
    build_env.add_argument(
        "-i",
        help=f"Path used by cmake to store compiled objects and exe (Default: {DEFAULT_BUILD_BASE}/<generator>/)",
        default=None,
    )

    parser.add_argument("-v", action="store_true", help="Activate verbose mode")

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
        else:
            goals[current_goal].append(arg)

    return args, goals


if __name__ == "__main__":
    args, goals = parse_args()

    for goal, goal_args in goals.items():
        run_goal(
            goal, GoalArgs(args.b, args.g, args.t, args.p, args.i, args.v, goal_args)
        )

    exit(0)
