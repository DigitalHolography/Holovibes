#!python

import os
import subprocess

if __name__ == "__main__":
    os.chdir("build/Ninja/Holovibes/")
    subprocess.call("ctest --verbose")