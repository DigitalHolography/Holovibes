project (CameraAlvium CXX)

add_library("${PROJECT_NAME}" SHARED
    camera_alvium.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${VIMBAX_SDK}/api/include"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${HOLO_DIR}/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    "${VIMBAX_SDK}/api/lib/VmbC.lib"
    "${VIMBAX_SDK}/api/lib/VmbCPP.lib"
    "${VIMBAX_SDK}/api/lib/VmbImageTransform.lib"

    Boost::boost
    spdlog::spdlog_header_only
)
