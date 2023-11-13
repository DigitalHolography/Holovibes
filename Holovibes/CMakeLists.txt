set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/")

# "--extended-lambda"                                      : enable __device__ lambda
# "-Xcudafe --diag_suppress=field_without_dll_interface"   : suppressed warning caused by Qt and NVCC conflict
string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda -Xcudafe --diag_suppress=field_without_dll_interface")

# "/W3"                     : Activate almost all warningd
# "/WX"                     : Transform all warnings into errors
string(APPEND CMAKE_CXX_FLAGS " /W3")

# ---------------------------------------------------------------------
# Includes and linkages
# ---------------------------------------------------------------------

# Packages

# Includes
# function(target_include_external_dirs target type)
#     foreach(dir ${ARGN})
#         target_compile_options(${target} ${type} "/external:I${dir}")
#     endforeach()
# endfunction()

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

# The following targets are going to be generated:
# - Holovibes_cuda_lib (lib)
# - Holovibes_lib (lib, inherits from Holovibes_cuda_lib)
# - Holovibes (executable, inherits from Holovibes_lib)
# - unit_tests (executable, inherits from Holovibes_lib)
#
# Holovibes_lib is the core containing all sources and linked to every external
# libraries (QT, Boost, ...). This way we can prevent redundancy and generate
# the main executable and tests based on it.


# ---------------------------------------------------------------------
# Target "${PROJECT_NAME}_cuda_lib"
# ---------------------------------------------------------------------

add_library("${PROJECT_NAME}_cuda_lib" STATIC)

set_property(TARGET "${PROJECT_NAME}_cuda_lib" PROPERTY CUDA_ARCHITECTURES OFF)
set_property(TARGET "${PROJECT_NAME}_cuda_lib" PROPERTY CUDA_STANDARD 17)

target_include_directories("${PROJECT_NAME}_cuda_lib" PRIVATE ${CUDA_EXTERNAL_INCLUDES} ${INTERNAL_INCLUDES})

target_link_libraries("${PROJECT_NAME}_cuda_lib" PRIVATE
    spdlog::spdlog_header_only
)

add_subdirectory(cuda_sources) # Add cuda sources to the target
# ---------------------------------------------------------------------
# Target "${PROJECT_NAME}_lib"
# ---------------------------------------------------------------------

add_library("${PROJECT_NAME}_lib" STATIC
     mainwindow.ui
     advancedsettingswindow.ui
     plotwindow.ui
)

target_include_directories("${PROJECT_NAME}_cuda_lib" INTERFACE ${EXTERNAL_INCLUDES})
target_include_directories("${PROJECT_NAME}_cuda_lib" INTERFACE ${INTERNAL_INCLUDES})

# Add sources to the target and includes that
# need to be compiled to the target
add_subdirectory(sources)
add_subdirectory(includes)

# Precompile header (to remove)
if(COMMAND target_precompile_headers)
    target_precompile_headers("${PROJECT_NAME}_lib" PUBLIC "includes/stdafx.hh")
endif()

target_link_libraries("${PROJECT_NAME}_lib" PUBLIC
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
    spdlog::spdlog_header_only
    "${PROJECT_NAME}_cuda_lib"
)

# Quick fix: https://stackoverflow.com/questions/44746308/how-to-make-include-directories-added-with-autouic-available-to-downstream-targe
add_custom_command(TARGET "${PROJECT_NAME}_lib" POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_lib_autogen/include/"
        "${CMAKE_CURRENT_BINARY_DIR}/Holovibes_autogen/include/"
)

# ---------------------------------------------------------------------
# Target Holovibes
# ---------------------------------------------------------------------

add_executable(Holovibes
    Holovibes.rc
    Holovibes.qrc
    sources/main.cc
)

target_link_libraries(Holovibes PRIVATE "${PROJECT_NAME}_lib")

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
    COMMAND Qt6::windeployqt $<TARGET_FILE_DIR:Holovibes>
)
