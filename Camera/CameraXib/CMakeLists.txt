project (CameraXib CXX)

add_library("${PROJECT_NAME}" SHARED
    camera_xib.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/libs/ximea/include/"
    "${CMAKE_SOURCE_DIR}/Holovibes/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/libs/ximea/xiapi64.lib"
    Boost::boost
    spdlog::spdlog_header_only
)