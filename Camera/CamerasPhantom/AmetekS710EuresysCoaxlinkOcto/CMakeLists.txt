project (AmetekS710EuresysCoaxlinkOcto CXX CUDA)

add_library("${PROJECT_NAME}" STATIC
    camera_phantom_s710.cc
    camera_phantom_s710_export.cc
    ../../Camera/camera_logger.cc
    ../camera_phantom_interface.cc
    ../camera_param_map.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${EURESYS_SDK}/include/"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Holovibes/includes/"
    "${CMAKE_SOURCE_DIR}/Camera/CamerasPhantom"
)

target_link_libraries("${PROJECT_NAME}"
    Boost::boost
    CUDA::cudart
    spdlog::spdlog_header_only
)
