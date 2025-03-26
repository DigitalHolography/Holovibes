project (CameraASI CXX CUDA)

add_library("${PROJECT_NAME}" SHARED
    camera_asi.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${ASI_SDK}/include"
    "${CAMERA_DIR}/Camera/"
    "${CAMERA_DIR}/include/"
    "${HOLO_DIR}/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    Boost::boost
    CUDA::cudart
    spdlog::spdlog_header_only
    "${ASI_SDK}/lib/x64/ASICamera2.lib"
)
