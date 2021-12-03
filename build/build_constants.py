import os
DEFAULT_GENERATOR = "Ninja"
DEFAULT_BUILD_MODE = "Debug"
DEFAULT_BUILD_BASE = "bin"
DEFAUT_TOOLCHAIN_FILE = "clang-cl-toolchain.cmake"

RUN_BINARY_FILE = "Holovibes.exe"

RELEASE_OPT = ["Release", "release", "R", "r"]
DEBUG_OPT = ["Debug", "debug", "D", "d"]
NINJA_OPT = ["Ninja", "ninja", "N", "n"]
NMAKE_OPT = ["NMake", "nmake", "NM", "nm"]
MAKE_OPT = ["Make", "make", "M", "m"]

CLANG_CL_OPT = ["clang-cl", "ClangCL", "clangcl", "Clang-cl", "Clang-CL"]
CL_OPT = ["cl", "CL", "MSVC", "msvc"]

INSTALLER_OUTPUT = "Output"
ISCC_FILE_TEMPLATE = "setupCreator.iss.jinja"
ISCC_FILE = "setupCreator.iss"
LIBS_PATH_FILE = "paths.json"
