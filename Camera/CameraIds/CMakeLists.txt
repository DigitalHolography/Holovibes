project (CameraIds CXX)

add_library("${PROJECT_NAME}" STATIC
    camera_ids.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/libs/ids/include/"
    "${CMAKE_SOURCE_DIR}/Holovibes/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/libs/ids/uEye_api_64.lib"
    Boost::boost
    spdlog::spdlog_header_only
)
