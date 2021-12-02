import os
import json
import subprocess

from .build_constants import *

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
    elif arg in MAKE_OPT:
        return "Unix Makefiles"
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


def get_build_dir(arg, generator):
    return arg or os.path.join(DEFAULT_BUILD_BASE, generator)


def cannot_find_vcvars():
    print("Cannot find the Developer Prompt launcher, you can either:")
    print("    - Find by yourself the vcvars64.bat file in your Visual Studio install")
    print("      Then specify it with the '-e' option")
    print("    - Find by yourself your Visual Studio install")
    print("      Then fill the env variable 'VS2019INSTALLDIR' with the path to it")
    exit(1)


def find_vcvars_auto():
    cmd = ['vswhere.exe', '-format', 'json', '-utf8', '-soft']
    pop = subprocess.check_call(cmd, stdout=subprocess.PIPE)

    for vs_install in json.loads(pop.stdout.read()):
        if not vs_install.get("installationPath"):
            continue

        vsvars = os.path.join(vs_install.get(
            "installationPath"), 'VC', 'Auxiliary', 'Build', 'vcvars64.bat')
        if os.path.isfile(vsvars):
            return vsvars

    raise Exception("Cannot find vcvars64.bat")


def find_vcvars():
    try:
        return find_vcvars_auto()
    except Exception:
        pass

    return find_vcvars_manual()


def find_vcvars_manual():
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
