project (CameraPhantomAutoDetection CXX CUDA)

add_library("${PROJECT_NAME}" SHARED
    camera_phantom_detection.cc
    ../Camera/camera_logger.cc
    camera_phantom_interface.cc
    camera_param_map.cc
    AmetekS710EuresysCoaxlinkOcto/camera_phantom_s710.cc
    AmetekS711EuresysCoaxlinkQsfp+/camera_phantom_s711.cc
    AmetekS991EuresysCoaxlinkQsfp+/camera_phantom_s991.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "AmetekS710EuresysCoaxlinkOcto/"
    "AmetekS711EuresysCoaxlinkQsfp+/"
    "AmetekS991EuresysCoaxlinkQsfp+/"
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


add_subdirectory(AmetekS710EuresysCoaxlinkOcto)
add_subdirectory(AmetekS711EuresysCoaxlinkQsfp+)
add_subdirectory(AmetekS991EuresysCoaxlinkQsfp+)