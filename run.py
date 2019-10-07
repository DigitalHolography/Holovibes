#!python

import os
import sys
import subprocess
import glob

class Config:
    conf = "Debug"
    gen = None

release_opt = ["Release", "release", "R", "r"]
debug_opt = ["Debug", "debug", "D", "d"]
ninja_opt = ["Ninja", "ninja", "N", "n"]
nmake_opt = ["NMake", "nmake", "NM", "nm"]
vs_opt = ["Visual Studio 14", "Visual Studio 15", "Visual Studio 16"]

def parse_args(config):
    for i in range(1, len(sys.argv)):
        arg = sys.argv[i]
        if arg in release_opt:
            config.conf = "Release"
        elif arg in debug_opt:
            config.conf = "Debug"
        elif arg in ninja_opt:
            config.gen = "Ninja"
        elif arg in nmake_opt:
            config.gen = "NMake Makefiles"
        elif arg in vs_opt:
            config.gen = arg
        else:
            break

def log(string, arg):
    print(f"[RUN.PY] {string}: {arg}", flush=True)

if __name__ == "__main__":
    config = Config()
    parse_args(config)

    log("CONFIG", config.conf)
    log("GENERATOR", config.gen)

    to_exec = ""
    if config.gen is None:
        files = glob.glob(f"build/**/{config.conf}/Holovibes.exe")
        if (len(files) == 0):
            log("ERROR", "Could not find any Holovibes.exe\nExiting")
            exit(1)
        log("FOUND", files)
        to_exec = sorted(files, key=lambda f: os.path.getmtime(f), reverse=True)[0]
    else:
        to_exec = f"build/{config.gen}/{config.conf}/Holovibes.exe"

    if not os.path.exists(to_exec):
        log("ERROR", f"File \"{to_exec}\" does not exist\nExiting")
        exit(1)

    log("EXEC", to_exec)
    os.chdir(os.path.dirname(to_exec))
    subprocess.call("Holovibes.exe")
