cmake_minimum_required (VERSION 3.0)

project (Cameraids CXX)

find_package(Boost REQUIRED)

add_library("${PROJECT_NAME}" SHARED
    camera_ids.cc
)

set(BITFLOW_SDK "C:/BitFlow SDK 6.30/")

include_directories(
    "${Boost_INCLUDE_DIRS}"

    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/lib/ids/include/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/lib/ids/uEye_api_64.lib"
)
