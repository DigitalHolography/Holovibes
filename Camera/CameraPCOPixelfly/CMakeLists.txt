cmake_minimum_required (VERSION 3.0)

project (CameraPCOPixelfly CXX)

add_library("${PROJECT_NAME}" SHARED
    camera_pco_pixelfly.cc
    "${CMAKE_SOURCE_DIR}/Camera/CameraPCO/camera_pco.cc"
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/Camera/CameraPCO/"
    "${CMAKE_SOURCE_DIR}/CameraUtils/"
    "${CMAKE_SOURCE_DIR}/lib/pco/include/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/lib/pco/SC2_Cam.lib"

    Boost::boost
)

# Generate the dll file in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)