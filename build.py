#!python

import os
import os.path as path
import sys
import subprocess

class Config:
    conf = "Debug"
    gen = "Ninja"
    remain = []
    remain_str = ""
    build_dir = "build/"

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
            config.remain = sys.argv[i:]
            config.remain_str = " ".join(config.remain)
            break
    config.build_dir += f"{config.gen}/"

def log(string, arg):
    print(f"[BUILD.PY] {string}: {arg}", flush=True)

if __name__ == "__main__":
    config = Config()
    parse_args(config)

    log("CONFIG", config.conf)
    log("GENERATOR", config.gen)
    log("REMAIN", config.remain)
    log("BUILD_DIR", config.build_dir)

    if "Visual Studio" in config.gen:
        if not os.path.isdir(config.build_dir): # create build dir
            cmd = ["cmake", "-G", config.gen, "-B", config.build_dir, "-S", ".", "-A", "x64"]
            log("CMD", cmd)
            subprocess.call(cmd)
        cmd = ["cmake", "--build", config.build_dir, "--config", config.conf] + config.remain + ["--", "/verbosity:normal"]
        log("CMD", cmd)
        subprocess.call(cmd)
    else:
        cmd = ["cmd.exe", "/c", "call", "C:\\Program Files (x86)\\Microsoft Visual Studio\\2017\\Community\\VC\\Auxiliary\\Build\\vcvars64.bat", "&&"]
        if not os.path.isdir(path.join(config.build_dir, config.conf)): # if build dir doesn't exist, run CMake configure step
            cmd += ["cmake", "-B", config.build_dir, "-S", ".", "-G", config.gen, f"-DCMAKE_BUILD_TYPE={config.conf}", "-DCMAKE_VERBOSE_MAKEFILE=ON", "&&"]
        cmd += ["cmake", "--build", config.build_dir] + config.remain
        log("CMD", cmd)
        subprocess.call(cmd)
