project(CameraPCO CXX)

add_library("${PROJECT_NAME}" SHARED
    pco_edge4.2lt.cc
    ../Camera/camera_logger.cc
    
    "${PCO_SDK}/pco.camera/camera.cpp"
    "${PCO_SDK}/pco.camera/cameraexception.cpp"
    "${PCO_SDK}/pco.camera/image.cpp"
    "${PCO_SDK}/pco.camera/xcite.cpp"
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

# === ADD THIS SECTION TO COPY PCO DLLs ===
# Find the PCO runtime DLLs
find_file(PCO_CONV_DLL "PCO_CONV.dll" PATHS "${PCO_SDK}/bin" NO_DEFAULT_PATH)
find_file(PCO_RECORDER_DLL "pco_recorder.dll" PATHS "${PCO_SDK}/bin" NO_DEFAULT_PATH)
find_file(SC2_CAM_DLL "SC2_Cam.dll" PATHS "${PCO_SDK}/bin" NO_DEFAULT_PATH)

# Copy DLLs to output directory
if(PCO_CONV_DLL AND PCO_RECORDER_DLL AND SC2_CAM_DLL)
    add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy "${PCO_CONV_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMAND "${CMAKE_COMMAND}" -E copy "${PCO_RECORDER_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMAND "${CMAKE_COMMAND}" -E copy "${SC2_CAM_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
        COMMENT "Copying PCO runtime DLLs to output directory"
    )
else()
    message(WARNING "Some PCO runtime DLLs not found in ${PCO_SDK}/bin/")
    if(NOT PCO_CONV_DLL)
        message(WARNING "Missing: PCO_CONV.dll")
    endif()
    if(NOT PCO_RECORDER_DLL)
        message(WARNING "Missing: pco_recorder.dll")
    endif()
    if(NOT SC2_CAM_DLL)
        message(WARNING "Missing: SC2_Cam.dll")
    endif()
endif()

# Optional: Also copy any additional required DLLs you might need
find_file(PCO_FILE_DLL "pco_file.dll" PATHS "${PCO_SDK}/bin" NO_DEFAULT_PATH)
if(PCO_FILE_DLL)
    add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -E copy "${PCO_FILE_DLL}" "$<TARGET_FILE_DIR:${PROJECT_NAME}>"
    )
endif()