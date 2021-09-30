#!python

import os
import sys
import subprocess
import argparse

DEFAULT_GENERATOR = "Ninja"
DEFAULT_BUILD_MODE = "Debug"
DEFAULT_GOAL = "build"

GOALS = ["cmake", "build", "run", "pytest", "ctest"]

RELEASE_OPT = ["Release", "release", "R", "r"]
DEBUG_OPT = ["Debug", "debug", "D", "d"]
NINJA_OPT = ["Ninja", "ninja", "N", "n"]
NMAKE_OPT = ["NMake", "nmake", "NM", "nm"]
VS_OPT = ["Visual Studio 14", "Visual Studio 15", "Visual Studio 16"]

#----------------------------------#
# Utils                            #
#----------------------------------#


def get_generator(arg):
    if not arg:
        return DEFAULT_GENERATOR
    elif arg in NINJA_OPT:
        return "Ninja"
    elif arg in NMAKE_OPT:
        return "NMake Makefiles"
    elif arg in VS_OPT:
        return arg
    else:
        raise Exception("Unreachable statement thanks to argparse")

def get_build_mode(arg):
    if not arg:
        return DEFAULT_BUILD_MODE
    elif arg in RELEASE_OPT:
        return "Release"
    elif arg in DEBUG_OPT:
        return "Debug"
    else:
        raise Exception("Unreachable statement thanks to argparse")


def cannot_find_vcvars():
    print("Cannot find the Developer Prompt launcher, you can either:")
    print("    - Find by yourself the vcvars64.bat file in your Visual Studio install")
    print("      Then specify it with the '-e' option")
    print("    - Find by yourself your Visual Studio install")
    print("      Then fill the env variable 'VS2019INSTALLDIR' with the path to it")
    exit(1)


def find_vcvars():
    parent2 = os.environ.get('VS2019INSTALLDIR')
    if not parent2 or not os.path.isdir(parent2):
        parent2 = os.environ.get('VS2017INSTALLDIR')
    if not parent2 or not os.path.isdir(parent2):
        parent1 = os.path.join('C:', 'Program Files (x86)',
                               'Microsoft Visual Studio', '2019')
        if not os.path.isdir(parent1):
            cannot_find_vcvars()

        parent2 = os.path.join(parent1, 'Professional')
        if not os.path.isdir(parent2):
            parent2 = os.path.join(parent1, 'Community')
        if not os.path.isdir(parent2):
            cannot_find_vcvars()

    res = os.path.join(parent2, 'VC', 'Auxiliary', 'Build', 'vcvars64.bat')
    if not os.path.isfile(res):
        cannot_find_vcvars()

    return "{}".format(res)

#----------------------------------#
# Goals                            #
#----------------------------------#


def cmake(args):
    cmd = ['cmd.exe', '/c', 'call']
    cmd += [args.e or find_vcvars(), '&&']

    generator = get_generator(args.g)
    build_mode = get_build_mode(args.b)
    build_dir = args.p or os.path.join('build', generator)

    # if build dir exist, remove it
    if os.path.isdir(build_dir):
        print("Warning: deleting previous build")
        sys.stdout.flush()
        subprocess.call(['rm', '-rf', build_dir], shell=True)

    cmd += ['cmake', '-B', build_dir,
            '-G', generator,
            '-S', '.',
            '-DCMAKE_VERBOSE_MAKEFILE=OFF',
            f'-DCMAKE_BUILD_TYPE={build_mode}',
            ]

    if args.g in VS_OPT:
        cmd += ['-A', 'x64']

    if args.v:
        print("Configure cmd: {}".format(' '.join(cmd)))
        sys.stdout.flush()

    return subprocess.call(cmd)


def build(args):
    build_mode = get_build_mode(args.b)
    build_dir = args.p or os.path.join('build', get_generator(args.g))

    if not os.path.isdir(build_dir):
        print("Build directory not found, Running configure goal before build")
        run_goal("cmake", args)

    cmd = ['cmd.exe', '/c', 'call']
    cmd += [args.e or find_vcvars(), '&&']
    cmd += ['cmake', '--build', build_dir]

    if args.g in VS_OPT:
        cmd += ['--config', build_mode,
                '--', '/verbosity:normal'
                ]

    if args.v:
        print("Build cmd: {}".format(' '.join(cmd)))
        sys.stdout.flush()

    return subprocess.call(cmd)


def run(args):
    build_mode = get_build_mode(args.b)
    exe_path = args.p or os.path.join(
        'build', get_generator(args.g), build_mode)
    previous_path = os.getcwd()

    if not os.path.isdir(exe_path):
        print("Cannot find Holovibes.exe at path: " + exe_path)
        sys.stdout.flush()
        exit(1)

    os.chdir(exe_path)

    cmd = ["Holovibes.exe", ]

    if args.v:
        print("Run cmd: {}".format(' '.join(cmd)))
        sys.stdout.flush()

    out = subprocess.call(cmd)

    os.chdir(previous_path)
    return out


def pytest(args):
    try:
        import pytest
    except ImportError as e:
        print(e)
        print("Please install pytest with '$ python -m pip install pytest'")
        sys.stdout.flush()

    if args.v:
        print("Pytest: Running pytest main...")
        sys.stdout.flush()

    return pytest.main(args=['-v', ])


def ctest(args):
    exe_path = args.p or os.path.join(
        'build', get_generator(args.g), "Holovibes")
    previous_path = os.getcwd()

    os.chdir(exe_path)
    cmd = ['cmd.exe', '/c', 'call', args.e or find_vcvars(), '&&' 'ctest',
           '--verbose']

    if args.v:
        print("Ctest cmd: {}".format(' '.join(cmd)))
        sys.stdout.flush()

    out = subprocess.call(cmd)

    os.chdir(previous_path)
    return out


def run_goal(goal: str, args) -> int:

    GoalsFuncs = {
        "cmake": cmake,
        "build": build,
        "run": run,
        "pytest": pytest,
        "ctest": ctest
    }

    goal_func = GoalsFuncs.get(goal)
    if not goal_func:
        raise Exception(f"Goal {goal} does not exists")

    out = goal_func(args)
    if out != 0:
        print(f"Goal {goal} Failed, Abort")
        sys.stdout.flush()
        exit(out)

#----------------------------------#
# CLI                              #
#----------------------------------#


def parse_args():
    parser = argparse.ArgumentParser(
        description='Holovibes Dev Tools (only runnable from project root)')

    build_mode = parser.add_argument_group(
        'Build Mode', 'Choose between Release mode and Debug mode (Default: Debug)')
    build_mode.add_argument(
        '-b', choices=RELEASE_OPT + DEBUG_OPT, default=None)

    build_generator = parser.add_argument_group(
        'Build Generator', 'Choose between NMake, Visual Studio and Ninja (Default: Ninja)')
    build_generator.add_argument(
        '-g', choices=NINJA_OPT + NMAKE_OPT + VS_OPT, default=None)

    build_env = parser.add_argument_group('Build Environment')
    build_env.add_argument('-e', help='Path to find the VS Developer Prompt to use to build (Default: auto-find)', default=None)
    build_env.add_argument('-p', help='Path used by cmake to store compiled objects and exe (Default: build/<generator>/)', default=None)

    parser.add_argument('-v', action="store_true",
                        help="Activate verbose mode")

    parser.add_argument('goals', choices=GOALS,
                        nargs='*', default=DEFAULT_GOAL)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # Shenanigans of argparse, when there are no goals specified the default is used
    # but the default is not a list which is the case if we specify any goal manually
    if args.goals != DEFAULT_GOAL:  # manually specified goals
        for goal in args.goals:
            run_goal(goal, args)
    else:                           # if there is no goal specified
        run_goal("build", args)

    exit(0)
