project (CameraOpenCV CXX)

add_library("${PROJECT_NAME}" STATIC
    camera_opencv.cc
    internal_conversion.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/Holovibes/includes/"
    "${OpenCV_INCLUDE_DIRS}"
)

target_link_libraries("${PROJECT_NAME}"
    Boost::boost
    opencv_videoio
    spdlog::spdlog_header_only
)
