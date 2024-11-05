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
    includes/gui/
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

set(VIMBAX_SDK "C:/Program Files/Allied Vision/Vimba X")

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
    assets/ui/mainwindow.ui
    assets/ui/advancedsettingswindow.ui
    assets/ui/plotwindow.ui
    assets/ui/lightui.ui
    assets/icons/Holovibes.ico
    assets/icons/holovibes.rc
    assets/style/style.css
)

qt_add_resources(Holovibes "HolovibesRessources"
    PREFIX "/"
    FILES
    assets/icons/Holovibes.ico
    assets/icons/holovibes_logo.png
    assets/style/drop_down.png
    assets/style/style.css
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
    AUTOUIC_SEARCH_PATHS "assets/ui"
)

target_compile_options(Holovibes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    -Xcudafe --diag_suppress=field_without_dll_interface
    -diag-suppress 27
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
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/assets/shaders/" $<TARGET_FILE_DIR:Holovibes>/AppData/shaders/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/assets/convolution_kernels/" $<TARGET_FILE_DIR:Holovibes>/AppData/convolution_kernels/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/assets/input_filters/" $<TARGET_FILE_DIR:Holovibes>/AppData/input_filters/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/assets/json_patches_holofile/" $<TARGET_FILE_DIR:Holovibes>/assets/json_patches_holofile/

    # Copy camera libs
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ximea/xiapi64.dll" $<TARGET_FILE_DIR:Holovibes>/xiapi64.dll
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ids/uEye_api_64.dll" $<TARGET_FILE_DIR:Holovibes>/uEye_api_64.dll
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/hamamatsu/dcamapi.dll" $<TARGET_FILE_DIR:Holovibes>/dcamapi.dll

    # Copy icon and logo
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/assets/icons/Holovibes.ico" $<TARGET_FILE_DIR:Holovibes>/assets/icons/Holovibes.ico
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/assets/icons/holovibes_logo.png" $<TARGET_FILE_DIR:Holovibes>/assets/icons/holovibes_logo.png
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/assets/style/style.css" $<TARGET_FILE_DIR:Holovibes>/assets/style/style.css
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/assets/style/drop_down.png" $<TARGET_FILE_DIR:Holovibes>/assets/style/drop_down.png

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
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_DIR}/configs/alvium.ini" $<TARGET_FILE_DIR:Holovibes>/AppData/cameras_config/alvium.ini

    # Copy Preset files to AppData
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/registration.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/registration.json
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/OCT.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/OCT.json
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/doppler_8b_384_27.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/doppler_8b_384_27.json
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/doppler_8b_512_40.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/doppler_8b_512_40.json
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/preset.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/preset.json
    COMMAND ${CMAKE_COMMAND} -E copy "${PRESET_DIR}/FILE_ON_GPU.json" $<TARGET_FILE_DIR:Holovibes>/AppData/preset/FILE_ON_GPU.json
)

# ---------------------------------------------------------------------
# Install
# ---------------------------------------------------------------------
install(TARGETS Holovibes
    RUNTIME DESTINATION .
    COMPONENT application
)

install(DIRECTORY "${HOLO_DIR}/assets/json_patches_holofile/"
    DESTINATION assets/json_patches_holofile
    COMPONENT application
)

install(DIRECTORY "${HOLO_DIR}/assets/icons/"
    DESTINATION assets/icons
    COMPONENT application
)

install(DIRECTORY "${HOLO_DIR}/assets/style/"
    DESTINATION assets/style
    COMPONENT application
)

install(PROGRAMS "${CMAKE_SOURCE_DIR}/ProcessHoloFiles.ps1"
    DESTINATION scripts
    COMPONENT process
)

install(PROGRAMS "${CMAKE_SOURCE_DIR}/ProcessRegistration.ps1"
    DESTINATION scripts
    COMPONENT registration
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

if(HAVE_BFType AND HAVE_BiDef AND HAVE_BiApi AND HAVE_BFErApi)
    install(FILES $<TARGET_FILE_DIR:Holovibes>/BitflowCyton.dll
        DESTINATION .
        COMPONENT application
    )
endif()

if(HAVE_BFType AND HAVE_BiDef AND HAVE_BiApi)
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
        $<TARGET_FILE_DIR:Holovibes>/CameraPhantomAutoDetection.dll
        DESTINATION .
        COMPONENT application
    )
endif()

if(HAVE_VIMBAX)
    add_custom_command(TARGET Holovibes POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/VmbCPP.dll" $<TARGET_FILE_DIR:Holovibes>/VmbCPP.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/VmbC.dll" $<TARGET_FILE_DIR:Holovibes>/VmbC.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/GenApi_MD_VC141_v3_2_AVT.dll" $<TARGET_FILE_DIR:Holovibes>/GenApi_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/GCBase_MD_VC141_v3_2_AVT.dll" $<TARGET_FILE_DIR:Holovibes>/GCBase_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/XmlParser_MD_VC141_v3_2_AVT.dll" $<TARGET_FILE_DIR:Holovibes>/XmlParser_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/Log_MD_VC141_v3_2_AVT.dll" $<TARGET_FILE_DIR:Holovibes>/Log_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/NodeMapData_MD_VC141_v3_2_AVT.dll" $<TARGET_FILE_DIR:Holovibes>/NodeMapData_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/MathParser_MD_VC141_v3_2_AVT.dll" $<TARGET_FILE_DIR:Holovibes>/MathParser_MD_VC141_v3_2_AVT.dll
        )
    install(FILES
        $<TARGET_FILE_DIR:Holovibes>/CameraAlvium.dll
        $<TARGET_FILE_DIR:Holovibes>/VmbCPP.dll
        $<TARGET_FILE_DIR:Holovibes>/VmbC.dll
        $<TARGET_FILE_DIR:Holovibes>/GenApi_MD_VC141_v3_2_AVT.dll
        $<TARGET_FILE_DIR:Holovibes>/GCBase_MD_VC141_v3_2_AVT.dll
        $<TARGET_FILE_DIR:Holovibes>/XmlParser_MD_VC141_v3_2_AVT.dll
        $<TARGET_FILE_DIR:Holovibes>/Log_MD_VC141_v3_2_AVT.dll
        $<TARGET_FILE_DIR:Holovibes>/NodeMapData_MD_VC141_v3_2_AVT.dll
        $<TARGET_FILE_DIR:Holovibes>/MathParser_MD_VC141_v3_2_AVT.dll
        DESTINATION .
        COMPONENT application
    )
endif()

install(DIRECTORY $<TARGET_FILE_DIR:Holovibes>/windeployqt/
    DESTINATION .
    COMPONENT application
)

install(DIRECTORY $<TARGET_FILE_DIR:Holovibes>/AppData/
    DESTINATION AppData
    COMPONENT application
)

# ---------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------
set(INSTALL_DIRECTORY "Holovibes\\\\${PROJECT_VERSION}")

set(CPACK_GENERATOR "NSIS")

set(CPACK_COMPONENT_APPLICATION_DISPLAY_NAME "Holovibes")
set(CPACK_COMPONENT_PROCESS_DISPLAY_NAME "Process Holo Files")
set(CPACK_COMPONENT_REGISTRATION_DISPLAY_NAME "Process Registration")
set(CPACK_COMPONENTS_ALL application process registration)

set(CPACK_PACKAGE_INSTALL_DIRECTORY ${INSTALL_DIRECTORY})

set(CPACK_NSIS_MUI_ICON "${HOLO_DIR}/assets/icons/Holovibes.ico")
set(CPACK_NSIS_MUI_UNIICON "${HOLO_DIR}/assets/icons/Holovibes.ico")
set(CPACK_NSIS_INSTALLED_ICON_NAME "Holovibes.exe")
set(CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/../LICENSE")

# Add SetCompress off to NSIS script
set(CPACK_NSIS_EXTRA_PREINSTALL_COMMANDS "
    SetCompress off
")

# Set the names of both shortcuts this installer creates: one for Holovibes and one for the .ps1
# These variable names are arbitrary and don't mean anything.
set(SHORTCUT_NAME "$DESKTOP\\\\Holovibes ${PROJECT_VERSION}.lnk")
set(PSONE_SHORTCUT_NAME "$INSTDIR\\\\scripts\\\\Process Holo Files.lnk")
set(PSONE_SHORTCUT_SHORTCUT_NAME "$DESKTOP\\\\Process Holo Files ${PROJECT_VERSION}.lnk")
set(PSONE2_SHORTCUT_NAME "$INSTDIR\\\\scripts\\\\Process Registration.lnk")
set(PSONE2_SHORTCUT_SHORTCUT_NAME "$DESKTOP\\\\Process Registration ${PROJECT_VERSION}.lnk")
# Note that the project version variable here has the format MAJOR.MINOR.PATCH, which is fine and explicit.
# But to fit with what Michael does currently, it could could also be MAJORMINORPATCH, like so:
# ${CMAKE_PROJECT_VERSION_MAJOR}${CMAKE_PROJECT_VERSION_MINOR}${CMAKE_PROJECT_VERSION_PATCH}

# Creating the shortcut to Holovibes on the desktop.
set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
    \\\${If} \\\${SectionIsSelected} \\\${application}
        CreateShortCut \\\"${SHORTCUT_NAME}\\\" \\\"$INSTDIR\\\\Holovibes.exe\\\"
    \\\${EndIf}

    \\\${If} \\\${SectionIsSelected} \\\${process}
        CreateShortCut \\\"${PSONE_SHORTCUT_NAME}\\\" \\\"powershell.exe\\\" \'-ExecutionPolicy Bypass -File \\\"$INSTDIR\\\\scripts\\\\ProcessHoloFiles.ps1\\\"\' \\\"$INSTDIR\\\\assets\\\\icons\\\\Script.ico\\\" 0
        CreateShortCut \\\"${PSONE_SHORTCUT_SHORTCUT_NAME}\\\" \\\"${PSONE_SHORTCUT_NAME}\\\" \\\"$INSTDIR\\\\assets\\\\icons\\\\Script.ico\\\" 0
    \\\${EndIf}

    \\\${If} \\\${SectionIsSelected} \\\${registration}
        CreateShortCut \\\"${PSONE2_SHORTCUT_NAME}\\\" \\\"powershell.exe\\\" \'-ExecutionPolicy Bypass -File \\\"$INSTDIR\\\\scripts\\\\ProcessRegistration.ps1\\\"\' \\\"$INSTDIR\\\\assets\\\\icons\\\\Script.ico\\\" 0
        CreateShortCut \\\"${PSONE2_SHORTCUT_SHORTCUT_NAME}\\\" \\\"${PSONE2_SHORTCUT_NAME}\\\" \\\"$INSTDIR\\\\assets\\\\icons\\\\Script.ico\\\" 0
    \\\${EndIf}
")

# Removes the Desktop shortcut of Holovibes.
set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
    Delete \\\"${SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE_SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE_SHORTCUT_SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE2_SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE2_SHORTCUT_SHORTCUT_NAME}\\\"
")

include(CPack)
