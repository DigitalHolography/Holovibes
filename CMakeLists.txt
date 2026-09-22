cmake_minimum_required(VERSION 3.31)

if(NOT CMAKE_HOST_WIN32)
    message(FATAL_ERROR "Holovibes currently requires Windows and MSVC.")
endif()

# Set manifest features before project(), which invokes the vcpkg toolchain.
option(BUILD_TESTING "Build C++ unit tests" OFF)
option(HOLOVIBES_DEPENDENCIES_ONLY "Build a dependency smoke test without CUDA or the application" OFF)
option(HOLOVIBES_BUILD_CAMERAS "Build camera plugins" ON)
if(BUILD_TESTING OR HOLOVIBES_DEPENDENCIES_ONLY)
    list(APPEND VCPKG_MANIFEST_FEATURES tests)
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# The project uses headers, not C++ modules; avoid a redundant scan per source.
set(CMAKE_CXX_SCAN_FOR_MODULES OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(HOLOVIBES_CUDA_ARCHITECTURES "75;86;89" CACHE STRING "CUDA architectures to compile")
set(CMAKE_CUDA_ARCHITECTURES "${HOLOVIBES_CUDA_ARCHITECTURES}")

set(holovibes_languages C CXX)
if(NOT HOLOVIBES_DEPENDENCIES_ONLY)
    list(APPEND holovibes_languages CUDA)
endif()

project(Holovibes
	VERSION 14.13.3
	DESCRIPTION "Holovibes"
	LANGUAGES ${holovibes_languages}
)

include(CTest)
include(cmake/Dependencies.cmake)
if(HOLOVIBES_DEPENDENCIES_ONLY)
    enable_testing()
    add_subdirectory(cmake/smoke)
    return()
endif()

find_package(CUDAToolkit 13.0 REQUIRED)
if(CMAKE_CUDA_COMPILER_VERSION VERSION_LESS 13.0)
    message(FATAL_ERROR "The selected CUDA compiler must be 13.0 or newer. Run scripts/bootstrap.ps1 and configure a fresh build directory.")
endif()
include(cmake/Deployment.cmake)

# CUDA's CCCL headers require the conforming MSVC preprocessor, including
# when included by ordinary C++ sources. Forward the option through nvcc.
add_compile_options(
    "$<$<COMPILE_LANGUAGE:CXX>:/Zc:preprocessor>"
    "$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/Zc:preprocessor>"
)

# Export compile_commands.json so IDE can use it for code completion.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(HOLO_DIR "${CMAKE_SOURCE_DIR}/Backend")
set(UI_DIR "${CMAKE_SOURCE_DIR}/Ui")
set(CAMERA_DIR "${CMAKE_SOURCE_DIR}/Camera")
set(CAMERA_LIB_DIR "${CMAKE_SOURCE_DIR}/Camera/libs")
set(PRESET_DIR "${UI_DIR}/assets/preset")

# Setup individual targets
if(HOLOVIBES_BUILD_CAMERAS)
    add_subdirectory(Camera)
endif()
add_subdirectory(Backend)
add_subdirectory(Ui)
add_subdirectory(docs)
