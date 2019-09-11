cmake_minimum_required (VERSION 3.0)

project (CameraIds CXX)

find_package(Boost REQUIRED)

add_library("${PROJECT_NAME}" SHARED
    camera_ids.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${Boost_INCLUDE_DIRS}"

    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/lib/ids/include/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/lib/ids/uEye_api_64.lib"
)
