import os
import sys
import json
import subprocess
from typing import List

from .build_constants import *


#----------------------------------#
# Utils                            #
#----------------------------------#


def get_generator(arg: str) -> str:
    if not arg:
        return DEFAULT_GENERATOR
    elif arg in NINJA_OPT:
        return "Ninja"
    elif arg in NMAKE_OPT:
        return "NMake Makefiles"
    elif arg in MAKE_OPT:
        return "Unix Makefiles"
    else:
        raise Exception(f"Generator undefined: {arg}")


def get_build_mode(arg: str) -> str:
    if not arg:
        return DEFAULT_BUILD_MODE
    elif arg in RELEASE_OPT:
        return "Release"
    elif arg in DEBUG_OPT:
        return "Debug"
    else:
        raise Exception(f"build mode undefined: {arg}")


def get_build_dir(arg: str, generator: str) -> str:
    return arg or os.path.join(DEFAULT_BUILD_BASE, generator)


def _get_toolchain_path(file: str) -> str:
    return os.path.join(os.path.dirname(os.path.realpath(__file__)), "cmake", "toolchains", file)


def get_toolchain(arg: str) -> str:
    if not arg:
        return _get_toolchain_path(DEFAUT_TOOLCHAIN_FILE)
    elif arg in CLANG_CL_OPT:
        return _get_toolchain_path("clang-cl-toolchain.cmake")
    elif arg in CL_OPT:
        return _get_toolchain_path("cl-toolchain.cmake")
    else:
        raise Exception(f"Unknown toolchain: {arg}")


def cannot_find_vcvars() -> None:
    print("Cannot find the Developer Prompt launcher, you can either:")
    print("    - Find by yourself the vcvars64.bat file in your Visual Studio install")
    print("      Then specify it with the '-e' option")
    print("    - Find by yourself your Visual Studio install")
    print("      Then fill the env variable 'VS2019INSTALLDIR' with the path to it")
    exit(1)


def find_vcvars_manual() -> str:
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


def find_vcvars_auto() -> str:
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


def find_vcvars() -> str:
    try:
        return find_vcvars_auto()
    except Exception:
        pass

    return find_vcvars_manual()


def is_windows() -> bool:
    return sys.platform.startswith('win32') or sys.platform.startswith('cygwin')


def get_vcvars_start_cmd(env) -> List[str]:
    if not is_windows():
        print("Warning: using vcvars cmd in not-windows env")

    return ['cmd.exe', '/c', 'call', env or find_vcvars(), '&&']