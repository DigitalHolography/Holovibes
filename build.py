#!python

import os
import sys
import subprocess
import argparse
import pathlib

DEFAULT_GENERATOR = "Ninja"
DEFAULT_BUILD_MODE = "Debug"

RELEASE_OPT = ["Release", "release", "R", "r"]
DEBUG_OPT = ["Debug", "debug", "D", "d"]
NINJA_OPT = ["Ninja", "ninja", "N", "n"]
NMAKE_OPT = ["NMake", "nmake", "NM", "nm"]
VS_OPT = ["Visual Studio 14", "Visual Studio 15", "Visual Studio 16"]


def parse_args():
    parser = argparse.ArgumentParser(description='Build Holovibes')

    build_mode = parser.add_argument_group(
        'Build Mode', 'Choose between Release mode and Debug mode (Default)')
    build_mode.add_argument(
        '-b', choices=RELEASE_OPT + DEBUG_OPT, default=None)

    build_generator = parser.add_argument_group(
        'Build Generator', 'Choose between NMake, Visual Studio and Ninja (Default)')
    build_generator.add_argument(
        '-g', choices=NINJA_OPT + NMAKE_OPT + VS_OPT, default=None)

    build_env = parser.add_argument_group('Build Environment')
    build_env.add_argument('-e', type=pathlib.Path,
                           help='Where to find the VS Developper Prompt to use to build', default=None)
    build_env.add_argument('-p', type=pathlib.Path,
                           help='Where to store cmake and build data', default=None)

    parser.add_argument('-v', action="store_true",
                        help="Activate verbose mode")

    return parser.parse_args()


def cannot_find_vcvars():
    print("Cannot find the Developper Prompt launcher, you can either:")
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

    return "\"{}\"".format(res)


def create_command(args):
    cmd = ['/c', 'call']
    cmd += [args.e or find_vcvars(), '&&']

    if not args.g:
        generator = DEFAULT_GENERATOR
    elif args.g in NINJA_OPT:
        generator = "Ninja"
    elif args.g in NMAKE_OPT:
        generator = "NMake Makefiles"
    elif args.g in VS_OPT:
        generator = args.g
    else:
        raise Exception("Unreachable statement thanks to argparse")

    build_mode = args.b or DEFAULT_BUILD_MODE
    build_dir = args.p or os.path.join('build', generator)

    # if build dir doesn't exist, run CMake configure step
    if not os.path.isdir(build_dir):
        cmd += ['cmake', '-B', build_dir,
                '-G', generator,
                '-S', '.',
                '-DCMAKE_VERBOSE_MAKEFILE=OFF',
                f'-DCMAKE_BUILD_TYPE={build_mode}',
                ]

        if args.g in VS_OPT:
            cmd += ['-A', 'x64']

        cmd.append('&&')

    cmd += ['cmake', '--build', build_dir]

    if args.g in VS_OPT:
        cmd += ['--config', build_mode,
                '--', '/verbosity:normal'
                ]

    return cmd


if __name__ == '__main__':
    args = parse_args()
    cmd = create_command(args)

    if args.v:
        print("Launch: cmd.exe " + " ".join(cmd))
        sys.stdout.flush()

    os.execvp('cmd.exe', cmd)


#--------------------------------#
# Obsolete script                #
#--------------------------------#

class Config:
    conf = "Debug"  # Configuration of the build: debug (default) or relase.
    # Build system generator: Ninja (default), NMake or Visual Studio (14, 15, 16)
    gen = "Ninja"
    remain = []
    remain_str = ""
    build_dir = "build/"  # Build directory
    # Visual studio version: Community (default) or Professional
    vs_version = "Community"

def parse_args(config):
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]
        if arg in RELEASE_OPT:
            config.conf = "Release"
        elif arg in DEBUG_OPT:
            config.conf = "Debug"
        elif arg in NINJA_OPT:
            config.gen = "Ninja"
        elif arg in NMAKE_OPT:
            config.gen = "NMake Makefiles"
        elif arg in VS_OPT:
            config.gen = arg
        elif arg in community_opt:
            config.vs_version = "Community"
        elif arg in professional_opt:
            config.vs_version = "Professional"
        else:
            config.remain = sys.argv[i:]
            config.remain_str = " ".join(config.remain)
            break
    config.build_dir += f"{config.gen}/"

# Logger


def log(string, arg):
    print(f"[BUILD.PY] {string}: {arg}", flush=True)


# Main function
if __name__ == "__main__":
    parse_argsv2()
    exit()
    config = Config()
    parse_args(config)  # Get the build configuration

    log("CONFIG", config.conf)
    log("GENERATOR", config.gen)
    log("REMAIN", config.remain)
    log("BUILD_DIR", config.build_dir)

    if "Visual Studio" in config.gen:  # If the build system generator is Visual Studio
        if not os.path.isdir(config.build_dir):  # create build dir
            cmd = ["cmake", "-G", config.gen, "-B",
                   config.build_dir, "-S", ".", "-A", "x64"]
            log("CMD", cmd)
            subprocess.call(cmd)
        # Build command
        cmd = ["cmake", "--build", config.build_dir, "--config",
               config.conf] + config.remain + ["--", "/verbosity:normal"]
        log("CMD", cmd)
        exit(subprocess.call(cmd))
    else:  # If the build system generator is Ninja or NMake
        # Create environment
        cmd = ["cmd.exe", "/c", "call", "C:\\Program Files (x86)\\Microsoft Visual Studio\\2019\\" +
               config.vs_version + "\\VC\\Auxiliary\\Build\\vcvars64.bat", "&&"]
        # if build dir doesn't exist, run CMake configure step
        if not os.path.isdir(path.join(config.build_dir, config.conf)):
            cmd += ["cmake", "-B", config.build_dir, "-S", ".", "-G", config.gen,
                    f"-DCMAKE_BUILD_TYPE={config.conf}", "-DCMAKE_VERBOSE_MAKEFILE=OFF", "&&"]
        # Build command
        cmd += ["cmake", "--build", config.build_dir] + config.remain
        log("CMD", cmd)
        exit(subprocess.call(cmd))
    exit(0)
