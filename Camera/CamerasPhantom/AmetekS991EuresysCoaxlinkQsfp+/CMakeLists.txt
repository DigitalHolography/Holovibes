project (AmetekS991EuresysCoaxlinkQsfp+ CXX CUDA)

add_library("${PROJECT_NAME}" SHARED
    camera_phantom_s991.cc
    camera_phantom_s991_export.cc
    ../../Camera/camera_logger.cc
    ../camera_phantom_interface.cc
    ../camera_param_map.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${EURESYS_SDK}/include/"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${CAMERA_DIR}/Camera/"
    "${CAMERA_DIR}/include/"
    "${CAMERA_DIR}/CamerasPhantom"
    "${HOLO_DIR}/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    Boost::boost
    CUDA::cudart
    spdlog::spdlog_header_only
)
