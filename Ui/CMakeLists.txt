# ---------------------------------------------------------------------
# Libraries only used by the UI
# ---------------------------------------------------------------------
# Dependencies are declared in cmake/Dependencies.cmake.

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


# ---------------------------------------------------------------------
# The executable
# ---------------------------------------------------------------------
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS sources/*.cc)
file(GLOB_RECURSE HEADERS CONFIGURE_DEPENDS includes/*.hh)
file(GLOB_RECURSE CUDA_SOURCES CONFIGURE_DEPENDS cuda_sources/*.cu)
file(GLOB_RECURSE CUDA_HEADERS CONFIGURE_DEPENDS cuda_includes/*.cuh)

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
    Qt6::Network
    Qt6::Widgets
    OpenGL::GL
    holovibes_headers
    CUDA::cudart
    Backend
)

target_include_directories(Holovibes PRIVATE
    ${EXTERNAL_INCLUDES}
    ${INTERNAL_INCLUDES}
    ${CUDA_EXTERNAL_INCLUDES}
)

set_target_properties(Holovibes PROPERTIES
    CUDA_ARCHITECTURES "${HOLOVIBES_CUDA_ARCHITECTURES}"
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
    $<$<COMPILE_LANGUAGE:CXX>:
    _CRT_SECURE_NO_WARNINGS
    >
)

if(COMMAND target_precompile_headers)
    target_precompile_headers(Holovibes PUBLIC "includes/stdafx.hh")
endif()

# ---------------------------------------------------------------------
# ProcessHoloFiles executable wrapper
# ---------------------------------------------------------------------
if(WIN32)
    add_executable(ProcessHoloFilesLauncher
        "${CMAKE_SOURCE_DIR}/scripts/ProcessHoloFilesLauncher.cpp"
    )
    set_target_properties(ProcessHoloFilesLauncher PROPERTIES
        OUTPUT_NAME ProcessHoloFiles
        RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
    )
    target_compile_features(ProcessHoloFilesLauncher PRIVATE cxx_std_17)
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
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/assets/json_patches_settings/" ${CMAKE_BINARY_DIR}/assets/json_patches_settings/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${PRESET_DIR}/" ${CMAKE_BINARY_DIR}/AppData/preset/

    # Copy camera config files to AppData
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CAMERA_DIR}/configs/" ${CMAKE_BINARY_DIR}/AppData/cameras_config/
    COMMAND ${CMAKE_COMMAND} -E remove_directory ${CMAKE_BINARY_DIR}/AppData/cameras_config/bfml/
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CAMERA_DIR}/configs/bfml/" ${CMAKE_BINARY_DIR}/AppData/bfml/

    # Copy icon and logo
    COMMAND ${CMAKE_COMMAND} -E copy "${UI_DIR}/assets/icons/Holovibes.ico" ${CMAKE_BINARY_DIR}/assets/icons/Holovibes.ico
    COMMAND ${CMAKE_COMMAND} -E copy "${UI_DIR}/assets/icons/holovibes_logo.png" ${CMAKE_BINARY_DIR}/assets/icons/holovibes_logo.png
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${UI_DIR}/assets/style/" ${CMAKE_BINARY_DIR}/assets/style/

    # One Qt deployment is reused for the build directory and installer.
    COMMAND Qt6::windeployqt $<IF:$<CONFIG:Debug>,--debug,--release>
        --dir "${CMAKE_BINARY_DIR}/qt-runtime" "$<TARGET_FILE:Holovibes>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_BINARY_DIR}/qt-runtime" "$<TARGET_FILE_DIR:Holovibes>"
    VERBATIM
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

install(DIRECTORY "${HOLO_DIR}/assets/json_patches_settings/"
    DESTINATION assets/json_patches_settings
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

install(PROGRAMS
    "${CMAKE_SOURCE_DIR}/ProcessHoloFiles.ps1"
    "${CMAKE_SOURCE_DIR}/ProcessHoloFiles.bat"
    DESTINATION scripts
    COMPONENT process
)

if(TARGET ProcessHoloFilesLauncher)
    install(TARGETS ProcessHoloFilesLauncher
        RUNTIME DESTINATION scripts
        COMPONENT process
    )
endif()

get_property(camera_targets GLOBAL PROPERTY HOLOVIBES_CAMERA_TARGETS)
if(camera_targets)
    add_dependencies(Holovibes ${camera_targets})
    install(TARGETS ${camera_targets} RUNTIME DESTINATION . COMPONENT application)
endif()
holovibes_deploy_runtime(Holovibes ${camera_targets})

if(BUILD_TESTING)
    add_test(NAME cli.recorded_file
        COMMAND powershell.exe -NoProfile -ExecutionPolicy Bypass
            -File "${CMAKE_SOURCE_DIR}/scripts/test-cli.ps1" -Executable "$<TARGET_FILE:Holovibes>")
    set_tests_properties(cli.recorded_file PROPERTIES TIMEOUT 180)
endif()

install(DIRECTORY "${CMAKE_BINARY_DIR}/qt-runtime/"
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

# Set the names of both shortcuts this installer creates: one for Holovibes, one for the .ps1/.bat, and one for the .exe wrapper
# These variable names are arbitrary and don't mean anything.
set(SHORTCUT_NAME "$DESKTOP\\\\Holovibes ${PROJECT_VERSION}.lnk")
set(PSONE_SHORTCUT_NAME "$INSTDIR\\\\scripts\\\\Process Holo Files.lnk")
set(PSONE_SHORTCUT_SHORTCUT_NAME "$DESKTOP\\\\Process Holo Files ${PROJECT_VERSION}.lnk")
set(PSONE_EXE_SHORTCUT_NAME "$INSTDIR\\\\scripts\\\\Process Holo Files (exe).lnk")
set(PSONE_EXE_SHORTCUT_SHORTCUT_NAME "$DESKTOP\\\\Process Holo Files (exe) ${PROJECT_VERSION}.lnk")
# Note that the project version variable here has the format MAJOR.MINOR.PATCH, which is fine and explicit.
# But to fit with what Michael does currently, it could could also be MAJORMINORPATCH, like so:
# ${CMAKE_PROJECT_VERSION_MAJOR}${CMAKE_PROJECT_VERSION_MINOR}${CMAKE_PROJECT_VERSION_PATCH}

# Creating the shortcut to Holovibes on the desktop.
set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "
    \\\${If} \\\${SectionIsSelected} \\\${application}
        CreateShortCut \\\"${SHORTCUT_NAME}\\\" \\\"$INSTDIR\\\\Holovibes.exe\\\"
    \\\${EndIf}

    \\\${If} \\\${SectionIsSelected} \\\${process}
        CreateShortCut \\\"${PSONE_SHORTCUT_NAME}\\\" \\\"$INSTDIR\\\\scripts\\\\ProcessHoloFiles.bat\\\" \\\"\\\" \\\"$INSTDIR\\\\assets\\\\icons\\\\Script.ico\\\" 0
        CreateShortCut \\\"${PSONE_SHORTCUT_SHORTCUT_NAME}\\\" \\\"${PSONE_SHORTCUT_NAME}\\\" \\\"$INSTDIR\\\\assets\\\\icons\\\\Script.ico\\\" 0
        CreateShortCut \\\"${PSONE_EXE_SHORTCUT_NAME}\\\" \\\"$INSTDIR\\\\scripts\\\\ProcessHoloFiles.exe\\\" \\\"\\\" \\\"$INSTDIR\\\\assets\\\\icons\\\\Script.ico\\\" 0
        CreateShortCut \\\"${PSONE_EXE_SHORTCUT_SHORTCUT_NAME}\\\" \\\"${PSONE_EXE_SHORTCUT_NAME}\\\" \\\"$INSTDIR\\\\assets\\\\icons\\\\Script.ico\\\" 0
    \\\${EndIf}
")

# Removes the Desktop shortcut of Holovibes.
set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "
    Delete \\\"${SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE_SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE_SHORTCUT_SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE_EXE_SHORTCUT_NAME}\\\"
    Delete \\\"${PSONE_EXE_SHORTCUT_SHORTCUT_NAME}\\\"
")

include(CPack)
