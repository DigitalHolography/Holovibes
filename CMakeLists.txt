cmake_minimum_required(VERSION 3.0)

project(Holovibes)

find_package(Boost REQUIRED
    COMPONENTS
    program_options
)

add_subdirectory(Camera)
add_subdirectory(CameraUtils)
add_subdirectory(GPIB)
add_subdirectory(Holovibes)
