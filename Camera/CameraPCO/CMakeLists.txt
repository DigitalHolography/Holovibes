project(CameraPCO CXX)

add_library("${PROJECT_NAME}" SHARED
    pco_edge4.2lt.cc
    ../Camera/camera_logger.cc
    
    # ADD THESE PCO C++ SOURCE FILES
    "${PCO_SDK}/pco.camera/camera.cpp"
    "${PCO_SDK}/pco.camera/cameraexception.cpp"
    "${PCO_SDK}/pco.camera/image.cpp"           # If it exists
    "${PCO_SDK}/pco.camera/xcite.cpp"  # If it exists
    # Add any other .cpp files from pco.camera directory
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${PCO_SDK}/include"
    "${PCO_SDK}/pco.camera"
    "${CAMERA_DIR}/Camera/"
    "${CAMERA_DIR}/include/"
    "${HOLO_DIR}/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    "${PCO_SDK}/lib/pco_conv.lib"
    "${PCO_SDK}/lib/pco_recorder.lib"
    "${PCO_SDK}/lib/pco_file.lib"
    "${PCO_SDK}/lib/sc2_cam.lib"
    "${PCO_SDK}/lib/etc_xcite.lib"
    Boost::boost
    spdlog::spdlog_header_only
)