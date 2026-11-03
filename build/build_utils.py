import os
import subprocess
from .build_constants import *


def get_preset(mode: str) -> str:
    if mode in DEBUG_OPT:
        return "windows-debug"
    if mode in RELEASE_OPT:
        return "windows-release"
    if mode is None or mode in DEV_OPT:
        return "windows-dev"
    raise ValueError(f"Unknown build mode: {mode}")


def get_build_dir(arg: str = None, mode: str = None) -> str:
    if arg is not None:
        raise ValueError("-i is replaced by CMake presets; see docs/DEVELOPMENT.md")
    return os.path.join("out", "build", get_preset(mode))


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
