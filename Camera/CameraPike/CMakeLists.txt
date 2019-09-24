cmake_minimum_required (VERSION 3.0)

project (CameraPike CXX)

add_library("${PROJECT_NAME}" SHARED
    camera_pike.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${Boost_INCLUDE_DIRS}"

    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/lib/avt/include/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/lib/avt/FGCamera.lib"
)

# Generate the dll file in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)