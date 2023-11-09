from conans import ConanFile, CMake, tools
from conans.errors import ConanInvalidConfiguration
import os
import sys
import json

from build.build_constants import *
from build import build_utils


class HolovibesConan(ConanFile):
    name = "Holovibes"
    version = "11.8.4"
    license = "GPL3"
    author = "Read AUTHORS.md"
    url = "https://holovibes.com/"
    description = "Real-time hologram rendering made easy"
    topics = "gpu", "cpp", "computer-vision", "opengl"
    settings = "os", "compiler", "build_type", "arch"
    build_requires = "cmake/3.24.0", "ninja/1.10.2"
    generators = "cmake_paths", "cmake_find_package", "virtualrunenv"

    options = {
        "cmake_compiler": GCC_OPT + CLANG_CL_OPT + CL_OPT + [None],
        "cmake_generator": NINJA_OPT + NMAKE_OPT + MAKE_OPT
    }

    requires = (
        "doxygen/1.9.4",
        "qt/6.2.1",
        "boost/1.71.0",
        "glm/0.9.9.8",
        "gtest/1.10.0",
        "nlohmann_json/3.10.4",
        "opencv/4.5.3",
        "opengl/system",
        "zlib/1.2.12",  # needed to overwrite qt bad dependency
        "openssl/1.1.1q",  # needed to overwrite qt bad dependency
        "freetype/2.11.0",  # needed to overwrite qt bad dependency
        "libpng/1.6.40", # needed to overwrite opencv bad dependency
        "spdlog/1.10.0",
    )

    default_options = (
        "qt:shared=True",
        "qt:widgets=True",
        "qt:qtcharts=True",
        "cmake_compiler=None",
        "cmake_generator=Ninja",
        "spdlog:header_only=True",
    )

    _cmake = None

    def validate(self):
        if self.settings.os != "Windows":
            raise ConanInvalidConfiguration("Build only available on windows")

    def source(self):
        self.run("git clone https://github.com/DigitalHolography/Holovibes.git")

    def _output_dll_paths(self):
        res = dict()
        for dep in self.deps_cpp_info.deps:
            if '-' not in dep:
                res[dep] = self.deps_cpp_info[dep].rootpath

        path = os.path.realpath("../../" + INSTALLER_OUTPUT)
        with open(os.path.join(path, LIBS_PATH_FILE), 'w') as fp:
            json.dump(res, fp)

    def _configure_cmake(self):
        if self._cmake:
            return self._cmake

        generator = self.options.cmake_generator
        toolchain = self.options.cmake_compiler
        # build_type = "Debug" if self.settings.build_type == "Debug" else "Release"

        self._cmake = CMake(
            self, generator=build_utils.get_generator(generator))
        self._cmake.definitions["CMAKE_TOOLCHAIN_FILE"] = build_utils.get_toolchain(
            toolchain)
        # self._cmake.definitions["CMAKE_BUILD_TYPE"] = build_type
        return self._cmake

    def _pytest(self):
        try:
            import pytest
        except ImportError as e:
            print(e)
            print("Please install pytest with '$ python -m pip install pytest'")
            sys.stdout.flush()
            raise

        return pytest.main(args=["-v"])

    def build(self):
        cmake = self._configure_cmake()
        cmake.configure()
        cmake.build()

        if self.should_test:
            self._pytest()

        if self.should_install:
            self._output_dll_paths()
