project (CameraHamamatsu CXX)

add_library("${PROJECT_NAME}" SHARED
    camera_hamamatsu.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/libs/hamamatsu/include/"
    "${CMAKE_SOURCE_DIR}/Holovibes/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/libs/hamamatsu/dcamapi.lib"
    Boost::boost
    spdlog::spdlog_header_only
)

# Generate the dll file in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)
