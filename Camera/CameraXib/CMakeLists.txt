project (CameraXib CXX)

add_library("${PROJECT_NAME}" SHARED
    camera_xib.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CAMERA_DIR}/Camera/"
    "${CAMERA_DIR}/include/"
    "${CAMERA_LIB_DIR}/ximea/include/"
    "${HOLO_DIR}/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CAMERA_LIB_DIR}/ximea/xiapi64.lib"
    Boost::boost
    spdlog::spdlog_header_only
)
