cmake_minimum_required (VERSION 3.0)

project (gpib CXX)

find_package(Boost REQUIRED)

add_library("${PROJECT_NAME}" SHARED
    gpib_controller.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${Boost_INCLUDE_DIRS}"
    "${CMAKE_CURRENT_SOURCE_DIR}/"
    "${CMAKE_SOURCE_DIR}/lib/ni_gpib/include/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/lib/ni_gpib/nivisa64.lib"
    "${CMAKE_SOURCE_DIR}/lib/ni_gpib/ni4882.lib"
)

# Generate the dll file in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)
