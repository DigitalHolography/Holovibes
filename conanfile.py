from conans import ConanFile, CMake, tools
from conans.errors import ConanInvalidConfiguration
import os


class HolovibesConan(ConanFile):
    name = "Holovibes"
    version = "10.4"
    license = "GPL3"
    author = "Read AUTHORS.md"
    url = "https://holovibes.com/"
    description = "Real-time hologram rendering made easy"
    topics = "gpu", "cpp", "computer-vision", "opengl"
    settings = "os", "compiler", "build_type", "arch"
    generators = "cmake_paths", "cmake_find_package"

    requires = (
        "qt/6.2.1",
        "boost/1.71.0",
        "glm/0.9.9.8",
        "gtest/1.10.0",
        "nlohmann_json/3.10.4",
        "opencv/4.5.3",
        "opengl/system",
        "freetype/2.11.0",  # needed to overwrite qt bad dependency
    )

    default_options = (
        "qt:shared=True",
        "qt:widgets=True",
        "qt:qtcharts=True"
    )

    def validate(self):
        if self.settings.os != "Windows":
            raise ConanInvalidConfiguration("Build only available on windows")

    def source(self):
        self.run("git clone https://github.com/DigitalHolography/Holovibes.git")