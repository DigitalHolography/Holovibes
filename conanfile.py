from conan import ConanFile
from conan.tools.cmake import cmake_layout, CMakeToolchain

class HolovibesRecipe(ConanFile):
    settings = "os", "compiler", "build_type", "arch"
    generators = "CMakeDeps"

    default_options = {
        "qt/*:shared": True,
        "qt/*:widgets": True,
        "qt/*:qtcharts": True,
        "spdlog/*:header_only": True,
    }

    def requirements(self):
        self.requires("qt/6.6.0")
        self.requires("boost/1.83.0")
        self.requires("glm/0.9.9.8")
        self.requires("gtest/1.14.0")
        self.requires("nlohmann_json/3.11.2")
        self.requires("opencv/4.5.5")
        self.requires("opengl/system")
        self.requires("spdlog/1.12.0")

        # Override requirements to resolve dependencies
        # versions conflicts
        self.requires("freetype/2.13.2", override=True)
        self.requires("libpng/1.6.42", override=True)

    #def build_requirements(self):
        #self.tool_requires("doxygen/1.9.4")

    def layout(self):
        cmake_layout(self)
        self.folders.build = "build/bin"
    
    def generate(self):
        toolchain = CMakeToolchain(self, 'Ninja')
        toolchain.generate()