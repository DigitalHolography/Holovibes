# ---------------------------------------------------------------------
# Libraries only used by the UI
# ---------------------------------------------------------------------
find_package(Qt6 REQUIRED COMPONENTS Widgets Core Charts OpenGL OpenGLWidgets)
find_package(opengl_system REQUIRED)
find_package(glm REQUIRED)

# https://doc.qt.io/qt-6/qt-standard-project-setup.html
qt_standard_project_setup()

# Where to look for .ui files.
set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/")

set(CUDA_EXTERNAL_INCLUDES
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
    includes/cli/
    includes/gui/
    includes/gui/notifier/
    includes/gui/other/
    includes/gui/selections/
    includes/gui/selections/interfaces/
    includes/gui/selections/overlays/
    includes/gui/windows/
    includes/gui/windows/panels/
    includes/gui/windows/AdvancedSettingsWindowTools
    includes/gui/windows/AdvancedSettingsWindowTools/panels
    "${HOLO_DIR}/cuda_includes/"
    "${HOLO_DIR}/includes/"
    "${HOLO_DIR}/includes/api/"
    "${HOLO_DIR}/includes/compute/"
    "${HOLO_DIR}/includes/data_struct/"
    "${HOLO_DIR}/includes/core/"
    "${HOLO_DIR}/includes/enum/"
    "${HOLO_DIR}/includes/struct/"
    "${HOLO_DIR}/includes/io/"
    "${HOLO_DIR}/includes/io/input_file/"
    "${HOLO_DIR}/includes/io/output_file/"
    "${HOLO_DIR}/includes/thread/"
    "${HOLO_DIR}/includes/tools/"
    "${HOLO_DIR}/includes/tools/cuda_tools/"
    "${HOLO_DIR}/includes/tools/units/"
    "${HOLO_DIR}/includes/tools/check/"
    "${HOLO_DIR}/includes/tools/exception/"
    "${HOLO_DIR}/includes/tools/logger/"
    "${HOLO_DIR}/includes/tools/state/"
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
    opengl::opengl
    Backend
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
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${UI_DIR}/assets/shaders/" ${CMAKE_BINARY_DIR}/AppData/shaders/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${UI_DIR}/assets/convolution_kernels/" ${CMAKE_BINARY_DIR}/AppData/convolution_kernels/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${UI_DIR}/assets/input_filters/" ${CMAKE_BINARY_DIR}/AppData/input_filters/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/assets/json_patches_holofile/" ${CMAKE_BINARY_DIR}/assets/json_patches_holofile/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${PRESET_DIR}/" ${CMAKE_BINARY_DIR}/AppData/preset/

    # Copy camera libs
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_LIB_DIR}/ximea/xiapi64.dll" ${CMAKE_BINARY_DIR}/xiapi64.dll
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_LIB_DIR}/ids/uEye_api_64.dll" ${CMAKE_BINARY_DIR}/uEye_api_64.dll
    COMMAND ${CMAKE_COMMAND} -E copy "${CAMERA_LIB_DIR}/hamamatsu/dcamapi.dll" ${CMAKE_BINARY_DIR}/dcamapi.dll

    # Copy camera config files to AppData
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CAMERA_DIR}/configs/" ${CMAKE_BINARY_DIR}/AppData/cameras_config/
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/AppData/cameras_config/bfml/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CAMERA_DIR}/configs/bfml/" ${CMAKE_BINARY_DIR}/AppData/bfml/

    # Copy icon and logo
    COMMAND ${CMAKE_COMMAND} -E copy "${UI_DIR}/assets/icons/Holovibes.ico" ${CMAKE_BINARY_DIR}/assets/icons/Holovibes.ico
    COMMAND ${CMAKE_COMMAND} -E copy "${UI_DIR}/assets/icons/holovibes_logo.png" ${CMAKE_BINARY_DIR}/assets/icons/holovibes_logo.png
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${UI_DIR}/assets/style/" ${CMAKE_BINARY_DIR}/assets/style/

    # Deploy Qt
    COMMAND Qt6::windeployqt ${CMAKE_BINARY_DIR}/Holovibes.exe

    # Deploy Qt in windeployqt directory so that it can be used by the installer
    COMMAND Qt6::windeployqt --dir $<TARGET_FILE_DIR:Holovibes>/windeployqt ${CMAKE_BINARY_DIR}/Holovibes.exe
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

install(DIRECTORY "${UI_DIR}/assets/icons/"
    DESTINATION assets/icons
    COMPONENT application
)

install(DIRECTORY "${UI_DIR}/assets/style/"
    DESTINATION assets/style
    COMPONENT application
)

install(PROGRAMS "${CMAKE_SOURCE_DIR}/ProcessHoloFiles.ps1"
    DESTINATION scripts
    COMPONENT process
)

install(FILES
    ${CMAKE_BINARY_DIR}/CameraHamamatsu.dll
    ${CMAKE_BINARY_DIR}/CameraIds.dll
    ${CMAKE_BINARY_DIR}/CameraOpenCV.dll
    ${CMAKE_BINARY_DIR}/CameraXib.dll
    ${CMAKE_BINARY_DIR}/CameraXiq.dll
    ${CMAKE_BINARY_DIR}/dcamapi.dll
    ${CMAKE_BINARY_DIR}/uEye_api_64.dll
    ${CMAKE_BINARY_DIR}/xiapi64.dll
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
    install(FILES ${CMAKE_BINARY_DIR}/BitflowCyton.dll
        DESTINATION .
        COMPONENT application
    )
endif()

if(HAVE_BFType AND HAVE_BiDef AND HAVE_BiApi)
    install(FILES ${CMAKE_BINARY_DIR}/CameraAdimec.dll
        DESTINATION .
        COMPONENT application
    )
endif()

if (HAVE_EGrabber AND HAVE_EGrabbers)
    install(FILES
        ${CMAKE_BINARY_DIR}/AmetekS710EuresysCoaxlinkOcto.dll
        ${CMAKE_BINARY_DIR}/AmetekS711EuresysCoaxlinkQsfp+.dll
        ${CMAKE_BINARY_DIR}/AmetekS991EuresysCoaxlinkQsfp+.dll
        ${CMAKE_BINARY_DIR}/CameraPhantomAutoDetection.dll
        DESTINATION .
        COMPONENT application
    )
endif()

if(HAVE_VIMBAX)
    add_custom_command(TARGET Holovibes POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/VmbCPP.dll" ${CMAKE_BINARY_DIR}/VmbCPP.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/VmbC.dll" ${CMAKE_BINARY_DIR}/VmbC.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/GenApi_MD_VC141_v3_2_AVT.dll" ${CMAKE_BINARY_DIR}/GenApi_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/GCBase_MD_VC141_v3_2_AVT.dll" ${CMAKE_BINARY_DIR}/GCBase_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/XmlParser_MD_VC141_v3_2_AVT.dll" ${CMAKE_BINARY_DIR}/XmlParser_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/Log_MD_VC141_v3_2_AVT.dll" ${CMAKE_BINARY_DIR}/Log_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/NodeMapData_MD_VC141_v3_2_AVT.dll" ${CMAKE_BINARY_DIR}/NodeMapData_MD_VC141_v3_2_AVT.dll
        COMMAND ${CMAKE_COMMAND} -E copy "${VIMBAX_SDK}/bin/MathParser_MD_VC141_v3_2_AVT.dll" ${CMAKE_BINARY_DIR}/MathParser_MD_VC141_v3_2_AVT.dll
        )
    install(FILES
        ${CMAKE_BINARY_DIR}/CameraAlvium.dll
        ${CMAKE_BINARY_DIR}/VmbCPP.dll
        ${CMAKE_BINARY_DIR}/VmbC.dll
        ${CMAKE_BINARY_DIR}/GenApi_MD_VC141_v3_2_AVT.dll
        ${CMAKE_BINARY_DIR}/GCBase_MD_VC141_v3_2_AVT.dll
        ${CMAKE_BINARY_DIR}/XmlParser_MD_VC141_v3_2_AVT.dll
        ${CMAKE_BINARY_DIR}/Log_MD_VC141_v3_2_AVT.dll
        ${CMAKE_BINARY_DIR}/NodeMapData_MD_VC141_v3_2_AVT.dll
        ${CMAKE_BINARY_DIR}/MathParser_MD_VC141_v3_2_AVT.dll
        DESTINATION .
        COMPONENT application
    )
endif()

install(DIRECTORY ${CMAKE_BINARY_DIR}/windeployqt/
    DESTINATION .
    COMPONENT application
)

install(DIRECTORY ${CMAKE_BINARY_DIR}/AppData/
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
set(CPACK_COMPONENTS_ALL application process)

set(CPACK_PACKAGE_INSTALL_DIRECTORY ${INSTALL_DIRECTORY})

set(CPACK_NSIS_MUI_ICON "${UI_DIR}/assets/icons/Holovibes.ico")
set(CPACK_NSIS_MUI_UNIICON "${UI_DIR}/assets/icons/Holovibes.ico")
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
")

# Removes the Desktop shortcut of Holovibes.
set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
    Delete \\\"${SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE_SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE_SHORTCUT_SHORTCUT_NAME}\\\"
")

include(CPack)