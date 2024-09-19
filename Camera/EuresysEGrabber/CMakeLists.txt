project (EuresysEGrabber CXX CUDA)

add_library("${PROJECT_NAME}" SHARED
    euresys.cc
    ../Camera/camera_logger.cc
    ../Camera/camera_utils.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${EURESYS_SDK}/include/"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Holovibes/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    Boost::boost
    CUDA::cudart
    spdlog::spdlog_header_only
)
