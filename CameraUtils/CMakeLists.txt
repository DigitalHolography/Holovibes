cmake_minimum_required (VERSION 3.0)

project (CameraUtils LANGUAGES CXX CUDA)

add_library("${PROJECT_NAME}" SHARED
    utils.cc
)

set_source_files_properties(utils.cc PROPERTIES LANGUAGE CUDA)

target_include_directories("${PROJECT_NAME}"
    PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/"
)

# Generate the dll file in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)