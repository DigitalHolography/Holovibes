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

add_executable(Holovibes
    ${SOURCES}
    ${HEADERS}
    ${CUDA_SOURCES}
    ${CUDA_HEADERS}
    Holovibes.rc
    Holovibes.qrc
    mainwindow.ui
    advancedsettingswindow.ui
    plotwindow.ui
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
    CUDA::cuda_driver
)

target_include_directories(Holovibes PRIVATE 
    ${EXTERNAL_INCLUDES} 
    ${INTERNAL_INCLUDES}
    ${CUDA_EXTERNAL_INCLUDES}
)

set_target_properties(Holovibes PROPERTIES
    CUDA_ARCHITECTURES OFF
    CUDA_STANDARD 17
)

target_compile_options(Holovibes PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
        --extended-lambda
        -Xcudafe --diag_suppress=field_without_dll_interface
    >
    $<$<COMPILE_LANGUAGE:CXX>:
        /W3
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

# COPY ALL DLLS TO INSTALL DIR USING A REGEX
install(DIRECTORY "${CMAKE_BINARY_DIR}"
    DESTINATION .
    COMPONENT application
    FILES_MATCHING PATTERN "*.dll"
)

# ---------------------------------------------------------------------
# Packaging
# ---------------------------------------------------------------------
set(INSTALL_DIRECTORY "Holovibes/Holovibes${PROJECT_VERSION}")

set(CPACK_GENERATOR "NSIS")
set(CPACK_COMPONENTS_ALL_IN_ONE_PACKAGE ON)
set(CPACK_COMPONENT_APPLICATION_DISPLAY_NAME "Holovibes")
set(CPACK_COMPONENTS_ALL application)
set(CPACK_PACKAGE_INSTALL_DIRECTORY ${INSTALL_DIRECTORY})

# set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS
#     ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/shaders/" "$INSTDIR/shaders/"
#     ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/convolution_kernels/" "$INSTDIR/convolution_kernels/"
#     ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/input_filters/" "$INSTDIR/input_filters/"
#     ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/json_patches_holofile/" "$INSTDIR/json_patches_holofile/"
#     ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ximea/xiapi64.dll" "$INSTDIR/xiapi64.dll"
#     ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ids/uEye_api_64.dll" "$INSTDIR/uEye_api_64.dll"
#     ${CMAKE_COMMAND} -E copy "${LIB_DIR}/hamamatsu/dcamapi.dll" "$INSTDIR/dcamapi.dll"
#     ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/resources/" "$INSTDIR/resources/"
#     ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/Holovibes.ico" "$INSTDIR/Holovibes.ico"
#     ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/holovibes_logo.png" "$INSTDIR/holovibes_logo.png"
# )

include(CPack)
