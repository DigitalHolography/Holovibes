project(Holovibes CXX CUDA)

# ---------------------------------------------------------------------
# Options and utils
# ---------------------------------------------------------------------

# compile_commands.json for vscode
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Force cmake to generate MOC from *.ui files for Qt
cmake_policy(SET CMP0100 NEW)

# Qt file generators
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/")

### CUDA Flags

# "--extended-lambda"                                      : enable __device__ lambda
# "-Xcudafe --diag_suppress=field_without_dll_interface"   : suppressed warning caused by Qt and NVCC conflict
string(APPEND CMAKE_CUDA_FLAGS " --extended-lambda -Xcudafe --diag_suppress=field_without_dll_interface")

### C++ Flags

# "/W3"                     : Activate almost all warningd
# "/WX"                     : Transform all warnings into errors
string(APPEND CMAKE_CXX_FLAGS " /W3 /WX")

# ---------------------------------------------------------------------
# Includes and linkages
# ---------------------------------------------------------------------

# Packages
find_package(Qt6 REQUIRED COMPONENTS Widgets Core Charts OpenGL OpenGLWidgets)
find_package(opengl_system REQUIRED)
find_package(OpenCV REQUIRED)
find_package(GTest REQUIRED)
find_package(glm REQUIRED)
find_package(nlohmann_json REQUIRED)

# Includes
function(target_include_external_dirs target type)
    foreach(dir ${ARGN})
        target_compile_options(${target} ${type} "/external:I${dir}")
    endforeach()
endfunction()

SET(CUDA_EXTERNAL_INCLUDES
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${glm_INCLUDE_DIRS}
    ${nlohmann_json_INCLUDE_DIRS}
)

set(EXTERNAL_INCLUDES
    ${opencv_INCLUDE_DIRS}
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

add_subdirectory(cuda_sources) # Add cuda sources to the target
# ---------------------------------------------------------------------
# Target "${PROJECT_NAME}_lib"
# ---------------------------------------------------------------------

add_library("${PROJECT_NAME}_lib" STATIC
     # CMake will automatically generate C++ headers
     # (ui_xxx.h) from the following QT .ui files.
     # Add the relevent .hh in Holovibes/includes/CMakeList.txt

     mainwindow.ui
     advancedsettingswindow.ui
     plotwindow.ui
)

target_include_external_dirs("${PROJECT_NAME}_cuda_lib" INTERFACE ${EXTERNAL_INCLUDES})
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
    ${opencv_LIBRARIES}
    ${opengl_system_LIBRARY_DIRS}

    cufft
    cublas
    cusolver

    "${PROJECT_NAME}_cuda_lib"
)

# Quick fix: https://stackoverflow.com/questions/44746308/how-to-make-include-directories-added-with-autouic-available-to-downstream-targe
add_custom_command(TARGET "${PROJECT_NAME}_lib" POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_lib_autogen/include/"
        "${CMAKE_CURRENT_BINARY_DIR}/Holovibes_autogen/include/"
)

# On MSVC, the Boost headers contains #pragma directives pointing to the
# corresponding libraries, automatically linking them.
if(MSVC)
    target_link_directories("${PROJECT_NAME}_lib" PUBLIC "${Boost_LIBRARY_DIRS}")
else()
    target_link_libraries("${PROJECT_NAME}_lib" PUBLIC Boost::program_options)
endif(MSVC)

# ---------------------------------------------------------------------
# Target unit_tests (unit tests)
# ---------------------------------------------------------------------

set(SRC_TESTS
    unit_tests/test_example.cc
    unit_tests/test_queue.cc
    unit_tests/test_chart.cc
    unit_tests/test_reduce.cc
    unit_tests/test_map.cc
    unit_tests/test_batch_input_queue.cc
    unit_tests/test_subsample_frame.cc
    unit_tests/test_fast_updates_holder.cc
    unit_tests/test_micro_cache.cc
    # Add your test files here
)

enable_testing()
find_package(Threads)
if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND ${THREADS_FOUND})

    message(STATUS "TESTS: Unit tests enabled")
    set(TEST_NAME "gtests")

    # Create an executable with all sources and lib needed
    add_executable(${TEST_NAME} ${SRC_TESTS})

    target_link_libraries(${TEST_NAME} PRIVATE
        "${PROJECT_NAME}_lib"
        GTest::gtest
        GTest::gtest_main
    )

    set_target_properties(${TEST_NAME} PROPERTIES
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/")

    add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME} "--gtest_color=yes")
    set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 120)

else()
    message(STATUS "TESTS: Unit tests disabled in Release mode")
endif()



# ---------------------------------------------------------------------
# Target Holovibes
# ---------------------------------------------------------------------

add_executable("${PROJECT_NAME}"
    # mainwindow.ui
    # advancedsettingswindow.ui
    # plotwindow.ui

    Holovibes.rc
    Holovibes.qrc

    sources/main.cc
)

target_link_libraries("${PROJECT_NAME}" PRIVATE "${PROJECT_NAME}_lib")

# Generate Holovibes.exe in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)


# ---------------------------------------------------------------------
# Post build: Copy necessary files to output dir
# ---------------------------------------------------------------------

add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD
    # Copy shaders and convolution kernels
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/shaders/" "${BUILD_DIR}/shaders/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/convolution_kernels/"
                                               "${BUILD_DIR}/convolution_kernels/"

    # Copy camera libs
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ximea/xiapi64.dll" "${BUILD_DIR}/xiapi64.dll"
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ids/uEye_api_64.dll" "${BUILD_DIR}/uEye_api_64.dll"
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/hamamatsu/dcamapi.dll" "${BUILD_DIR}/dcamapi.dll"

    # Copy not all but some resources (python scripts, ...)
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CMAKE_SOURCE_DIR}/resources/instruction_file_samples/"
                                               "${BUILD_DIR}/instruction_file_samples/"

    # Copy icon and logo
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/Holovibes.ico" "${BUILD_DIR}/Holovibes.ico"
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/holovibes_logo.png" "${BUILD_DIR}/holovibes_logo.png"
)
