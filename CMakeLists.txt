cmake_minimum_required(VERSION 3.17)

project(Holovibes)

set(CONAN_CMAKE_SILENT_OUTPUT ON)

# Create all the FindXXX.cmake needed for the project
set(CONAN_CMAKE_SILENT_OUTPUT ON)
include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

set(LIB_DIR "${CMAKE_SOURCE_DIR}/libs")
set(HOLO_DIR "${CMAKE_SOURCE_DIR}/Holovibes")
set(CAMERA_DIR "${CMAKE_SOURCE_DIR}/Camera")
set(BUILD_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")

# Used in every sub project
find_package(Boost REQUIRED
	COMPONENTS
	program_options
)

add_subdirectory(Camera)
add_subdirectory(GPIB)
add_subdirectory(Holovibes)
