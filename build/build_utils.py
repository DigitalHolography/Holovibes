import os
import sys
import json
import subprocess
from typing import List

from .build_constants import *


#----------------------------------#
# Utils                            #
#----------------------------------#

def is_windows() -> bool:
    return sys.platform.startswith('win32') or sys.platform.startswith('cygwin')


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
        if is_windows():
            return _get_toolchain_path(DEFAUT_WIN64_TOOLCHAIN_FILE)
        else:
            return _get_toolchain_path(DEFAUT_LINUX_TOOLCHAIN_FILE)
    elif arg in CLANG_CL_OPT:
        return _get_toolchain_path("clang-cl-toolchain.cmake")
    elif arg in CL_OPT:
        return _get_toolchain_path("cl-toolchain.cmake")
    elif arg in GCC_OPT:
        return _get_toolchain_path("gcc-toolchain.cmake")
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
    cmd = [os.path.join(os.path.dirname(os.path.realpath(__file__)), "vswhere.exe"), '-format', 'json', '-utf8', '-sort']
    pop = subprocess.run(cmd, stdout=subprocess.PIPE)

    pop.check_returncode()

    for vs_install in json.loads(pop.stdout.decode('utf-8')):
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


def get_vcvars_start_cmd(env) -> List[str]:
    if not is_windows():
        print("Warning: using vcvars cmd in not-windows env")

    return ['cmd.exe', '/c', 'call', env or find_vcvars(), '&&']

def get_conan_venv_start_cmd(build_dir: str, generator: str=None):
    if not is_windows():
        print("Warning: using conan win venv cmd in not-windows env")

    venv_path = os.path.join(
        get_build_dir(
            build_dir, get_generator(generator)
        ),
        "activate_run.bat"
    )
    return ['cmd.exe', '/c', 'call', venv_path, '&&']

def get_lib_paths() -> str:
    with open(os.path.join(INSTALLER_OUTPUT, LIBS_PATH_FILE), 'r') as fp:
        return json.load(fp)


def bump_all_versions(type) -> str:
    try:
        return subprocess.call([
            "bump2version",
            type,
            '--allow-dirty',
            "--config-file", os.path.join(os.path.dirname(
                os.path.realpath(__file__)), ".bumpversion.cfg")
        ]
        )
    except:
        print("Please make sure you have installed the build/requirements.txt file")
        raise


def create_release_file(paths, build_dir: str):
    from jinja2 import Environment, FileSystemLoader
    env = Environment(loader=FileSystemLoader(
        os.path.join(os.path.dirname(os.path.realpath(__file__)))))
    template = env.get_template(ISCC_FILE_TEMPLATE)
    output_from_parsed_template = template.render(
        paths=paths, build_dir=build_dir, binary_filename=RUN_BINARY_FILE)

    # to save the results
    with open(ISCC_FILE, "w") as fh:
        fh.write(output_from_parsed_template)


def get_cmake_variable(build_dir: str, variable: str) -> str:
    cmd = f"cmake -B {build_dir} -LA | grep {variable} | cut -d '=' -f 2 -"
    return subprocess.check_output(cmd, shell=True).decode('utf-8')
