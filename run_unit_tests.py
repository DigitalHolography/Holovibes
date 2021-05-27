#!python

import os
import subprocess

if __name__ == "__main__":
    os.chdir("build/Ninja/Holovibes/")
    exit(subprocess.call("cmd.exe /c ctest --verbose", shell=True))
