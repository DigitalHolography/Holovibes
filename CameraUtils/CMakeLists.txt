cmake_minimum_required (VERSION 3.0)

project (CameraUtils CUDA CXX)

add_library("${PROJECT_NAME}" SHARED
    utils.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/"
)

target_link_libraries("${PROJECT_NAME}"
    cudart
)

# Generate the dll file in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)