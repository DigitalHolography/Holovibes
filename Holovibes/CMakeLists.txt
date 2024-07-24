# Where to look for .ui files.
set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/")

SET(CUDA_EXTERNAL_INCLUDES
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${glm_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
)

set(EXTERNAL_INCLUDES
    ${Boost_INCLUDE_DIRS}
    ${Qt6_INCLUDE_DIRS}
    ${CUDA_EXTERNAL_INCLUDES}
)

set(INTERNAL_INCLUDES
    cuda_includes/
    includes/
    includes/state/
    includes/compute/
    includes/core/
    includes/enum/
    includes/struct/
    includes/gui/other/
    includes/gui/selections/
    includes/gui/selections/interfaces/
    includes/gui/selections/overlays/
    includes/gui/windows/
    includes/gui/windows/panels/
    includes/gui/windows/AdvancedSettingsWindowTools
    includes/gui/windows/AdvancedSettingsWindowTools/panels
    includes/io_files/
    includes/thread/
    includes/tools/
    includes/tools/cuda_tools/
    includes/tools/units/
    includes/tools/check/
    includes/tools/exception/
    includes/tools/logger/
    "${CMAKE_SOURCE_DIR}/GPIB/"
    "${CAMERA_DIR}/include/"
)

# ---------------------------------------------------------------------
# The executable
# ---------------------------------------------------------------------
file(GLOB_RECURSE SOURCES sources/*.cc)
file(GLOB_RECURSE HEADERS includes/*.hh)
file(GLOB_RECURSE CUDA_SOURCES cuda_sources/*.cu)
file(GLOB_RECURSE CUDA_HEADERS cuda_includes/*.cuh)

qt_add_executable(Holovibes
    ${SOURCES}
    ${HEADERS}
    ${CUDA_SOURCES}
    ${CUDA_HEADERS}
    mainwindow.ui
    advancedsettingswindow.ui
    plotwindow.ui
    lightui.ui
    Holovibes.ico
    holovibes.rc
)

qt_add_resources(Holovibes "HolovibesRessources"
    PREFIX "/"
    FILES
    Holovibes.ico
    holovibes_logo.png
)

target_link_libraries(Holovibes PRIVATE 
    spdlog::spdlog_header_only
    Boost::program_options
    Qt6::OpenGL
    Qt6::OpenGLWidgets
    Qt6::Charts
    Qt6::Core
    Qt6::Gui
    Qt6::Widgets
    opencv_videoio
    opengl::opengl
    cufft
    cublas
    cusolver
    nvml
)

target_include_directories(Holovibes PRIVATE 
    ${EXTERNAL_INCLUDES} 
    ${INTERNAL_INCLUDES}
    ${CUDA_EXTERNAL_INCLUDES}
)

set_target_properties(Holovibes PROPERTIES
    CUDA_ARCHITECTURES "61;75;86;89"
    CUDA_STANDARD 17
)

target_compile_options(Holovibes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        -Xcudafe --diag_suppress=field_without_dll_interface
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        /W3
        /wd4005
        /wd4003
    >
)

target_compile_definitions(Holovibes PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:.
        _CRT_SECURE_NO_WARNINGS
    >
)

if(COMMAND target_precompile_headers)
    target_precompile_headers(Holovibes PUBLIC "includes/stdafx.hh")
endif()

# ---------------------------------------------------------------------
# Post build: Copy necessary files to output dir
# ---------------------------------------------------------------------
add_custom_command(TARGET Holovibes POST_BUILD
    # Copy shaders and convolution kernels and json patch
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/shaders/" $<TARGET_FILE_DIR:Holovibes>/shaders/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/convolution_kernels/" $<TARGET_FILE_DIR:Holovibes>/convolution_kernels/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/input_filters/" $<TARGET_FILE_DIR:Holovibes>/input_filters/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/json_patches_holofile/" $<TARGET_FILE_DIR:Holovibes>/json_patches_holofile/

    # Copy camera libs
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ximea/xiapi64.dll" $<TARGET_FILE_DIR:Holovibes>/xiapi64.dll
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ids/uEye_api_64.dll" $<TARGET_FILE_DIR:Holovibes>/uEye_api_64.dll
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/hamamatsu/dcamapi.dll" $<TARGET_FILE_DIR:Holovibes>/dcamapi.dll

    # Copy not all but some resources (python scripts, ...)
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/resources/" $<TARGET_FILE_DIR:Holovibes>/resources/

    # Copy icon and logo
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/Holovibes.ico" $<TARGET_FILE_DIR:Holovibes>/Holovibes.ico
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/holovibes_logo.png" $<TARGET_FILE_DIR:Holovibes>/holovibes_logo.png

    # Deploy Qt
    COMMAND Qt6::windeployqt $<TARGET_FILE_DIR:Holovibes>/Holovibes.exe

    # Deploy Qt in windeployqt directory so that it can be used by the installer
    COMMAND Qt6::windeployqt --dir $<TARGET_FILE_DIR:Holovibes>/windeployqt $<TARGET_FILE_DIR:Holovibes>/Holovibes.exe

    # Copy camera config files to AppData
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/adimec.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/adimec.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/bitflow.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/bitflow.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/hamamatsu.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/hamamatsu.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/ids.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/ids.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/opencv.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/opencv.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/ametek_s710_euresys_coaxlink_octo.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/ametek_s710_euresys_coaxlink_octo.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/ametek_s991_euresys_coaxlink_qsfp+.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/ametek_s991_euresys_coaxlink_qsfp+.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/ametek_s711_euresys_coaxlink_qsfp+.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/ametek_s711_euresys_coaxlink_qsfp+.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/xib.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/xib.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/xiq.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/xiq.ini
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/phantom.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/phantom.ini
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CAMERA_DIR}/configs/bfml/" $<TARGET_FILE_DIR:Holovibes>/AppData/bfml/

    # Copy Preset files to AppData
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/OCT.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/OCT.json
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/DOPPLER.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/DOPPLER.json
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/DOPPLER.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/preset.json
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/FILE_ON_GPU.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/FILE_ON_GPU.json

)

# ---------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------
install(TARGETS Holovibes
    RUNTIME DESTINATION .
    COMPONENT application
)

install(DIRECTORY "${HOLO_DIR}/shaders/"
    DESTINATION shaders
    COMPONENT application
)

install(DIRECTORY "${HOLO_DIR}/convolution_kernels/"
    DESTINATION convolution_kernels
    COMPONENT application
)

install(DIRECTORY "${HOLO_DIR}/input_filters/"
    DESTINATION input_filters
    COMPONENT application
)

install(DIRECTORY "${HOLO_DIR}/json_patches_holofile/"
    DESTINATION json_patches_holofile
    COMPONENT application
)

install(DIRECTORY "${CMAKE_SOURCE_DIR}/resources/"
    DESTINATION resources
    COMPONENT application
)

install(FILES "${HOLO_DIR}/Holovibes.ico"
    DESTINATION .
    COMPONENT application
)

install(FILES "${HOLO_DIR}/holovibes_logo.png"
    DESTINATION .
    COMPONENT application
)


install(FILES 
    $<TARGET_FILE_DIR:Holovibes>/CameraHamamatsu.dll
    $<TARGET_FILE_DIR:Holovibes>/CameraIds.dll
    $<TARGET_FILE_DIR:Holovibes>/CameraOpenCV.dll
    $<TARGET_FILE_DIR:Holovibes>/CameraXib.dll
    $<TARGET_FILE_DIR:Holovibes>/CameraXiq.dll
    $<TARGET_FILE_DIR:Holovibes>/dcamapi.dll
    $<TARGET_FILE_DIR:Holovibes>/uEye_api_64.dll
    $<TARGET_FILE_DIR:Holovibes>/xiapi64.dll
    ${CUDA_TOOLKIT_ROOT_DIR}/bin/cublas64_12.dll
    ${CUDA_TOOLKIT_ROOT_DIR}/bin/cublasLt64_12.dll
    ${CUDA_TOOLKIT_ROOT_DIR}/bin/cudart64_12.dll
    ${CUDA_TOOLKIT_ROOT_DIR}/bin/cufft64_11.dll
    ${CUDA_TOOLKIT_ROOT_DIR}/bin/cusolver64_11.dll
    ${CUDA_TOOLKIT_ROOT_DIR}/bin/cusparse64_12.dll
    ${CUDA_TOOLKIT_ROOT_DIR}/bin/nvJitLink_120_0.dll
    DESTINATION .
    COMPONENT application
)


if (HAVE_BFType AND HAVE_BiDef AND HAVE_BiApi AND HAVE_BFErApi)
    install(FILES $<TARGET_FILE_DIR:Holovibes>/BitflowCyton.dll
        DESTINATION .
        COMPONENT application
    )
endif()

if (HAVE_BFType AND HAVE_BiDef AND HAVE_BiApi)
    install(FILES $<TARGET_FILE_DIR:Holovibes>/CameraAdimec.dll
        DESTINATION .
        COMPONENT application
    )
endif()

if (HAVE_EGrabber AND HAVE_EGrabbers)
    install(FILES 
        $<TARGET_FILE_DIR:Holovibes>/AmetekS710EuresysCoaxlinkOcto.dll
        $<TARGET_FILE_DIR:Holovibes>/AmetekS711EuresysCoaxlinkQsfp+.dll
        $<TARGET_FILE_DIR:Holovibes>/AmetekS991EuresysCoaxlinkQsfp+.dll
        DESTINATION .
        COMPONENT application)
endif()

install(DIRECTORY $<TARGET_FILE_DIR:Holovibes>/windeployqt/
    DESTINATION .
    COMPONENT application
)

install(DIRECTORY $<TARGET_FILE_DIR:Holovibes>/AppData/
    DESTINATION "AppData/"
    COMPONENT application
)


# ---------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------
set(INSTALL_DIRECTORY "Holovibes\\\\${PROJECT_VERSION}")

set(CPACK_GENERATOR "NSIS")
set(CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE ON)
set(CPACK_COMPONENT_APPLICATION_DISPLAY_NAME "Holovibes")
set(CPACK_COMPONENTS_ALL application)
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${INSTALL_DIRECTORY})

set(CPACK_NSIS_MUI_ICON "${HOLO_DIR}/Holovibes.ico")
set(CPACK_NSIS_MUI_UNIICON "${HOLO_DIR}/Holovibes.ico")
set(CPACK_NSIS_INSTALLED_ICON_NAME "Holovibes.exe")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")

# Add SetCompress off to NSIS script
set(CPACK_NSIS_EXTRA_PREINSTALL_COMMANDS "
    SetCompress off
")

include(CPack)
