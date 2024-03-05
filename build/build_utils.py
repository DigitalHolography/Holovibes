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


def get_build_dir(arg: str) -> str:
    return os.path.join(DEFAULT_BUILD_FOLDER, DEFAULT_BUILD_BASE)


def get_conan_venv_start_cmd(build_dir: str, generator: str=None):
    if not is_windows():
        print("Warning: using conan win venv cmd in not-windows env")

    venv_path = os.path.join(os.getcwd(),
                            get_build_dir(build_dir, get_generator(generator)),
                            "activate_run.bat")
    
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
