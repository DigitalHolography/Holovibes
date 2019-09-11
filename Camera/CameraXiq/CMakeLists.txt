cmake_minimum_required (VERSION 3.0)

project (CameraXiq CXX)

find_package(Boost REQUIRED)

add_library("${PROJECT_NAME}" SHARED
    camera_xiq.cc
)

include_directories(
    "${Boost_INCLUDE_DIRS}"

    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/CameraUtils/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/lib/ximea/include/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/lib/ximea/m3apiX64.lib"
)
