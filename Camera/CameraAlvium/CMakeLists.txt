project (CameraAlvium CXX)

add_library("${PROJECT_NAME}" SHARED
    camera_alvium.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/libs/VmbCPP/"
    "${CMAKE_SOURCE_DIR}/Holovibes/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/libs/VmbCPP/VmbCPP.lib"
    "${CMAKE_SOURCE_DIR}/libs/VmbCPP/VmbC.lib"
    "${CMAKE_SOURCE_DIR}/libs/VmbCPP/VmbImageTransform.lib"
    Boost::boost
    spdlog::spdlog_header_only
)
