project (CameraIds CXX)

add_library("${PROJECT_NAME}" SHARED
    camera_ids.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CAMERA_DIR}/Camera/"
    "${CAMERA_DIR}/include/"
    "${CAMERA_LIB_DIR}/ids/include/"
    "${HOLO_DIR}/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CAMERA_LIB_DIR}/ids/uEye_api_64.lib"
    Boost::boost
    spdlog::spdlog_header_only
)
