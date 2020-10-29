project(Holovibes CXX CUDA)

# Packages
find_package(Qt5 COMPONENTS Widgets Core Charts REQUIRED)
find_package(OpenGL REQUIRED)

cmake_policy(SET CMP0100 NEW)

# Qt file generators
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/")

string(APPEND CMAKE_CUDA_FLAGS " --expt-extended-lambda")

set(HOLO_INCLUDES
    cuda_includes/
    includes/
    includes/core/
    includes/compute/
    includes/io_files/
    includes/thread/
    includes/tools/
    includes/tools/cuda_tools/
    includes/tools/units/
    includes/gui/other/
    includes/gui/selections/
    includes/gui/selections/interfaces/
    includes/gui/selections/overlays/
    includes/gui/windows/

    "${CMAKE_SOURCE_DIR}/libs/glm/"
    "${CMAKE_SOURCE_DIR}/GPIB/"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${CMAKE_SOURCE_DIR}/Camera/include/"
)



# The following targets are going to be generated:
# - Holovibes_cuda_lib (lib)
# - Holovibes_lib (lib, inherits from Holovibes_cuda_lib)
# - Holovibes (executable, inherits from Holovibes_lib)
# - test_* (executable, inherits from Holovibes_lib)
#
# Holovibes_lib is the core containing all sources and linked to every external
# libraries (QT, Boost, ...). This way we can prevent redundancy and generate
# the main executable and tests based on it.


# ---
# Target "${PROJECT_NAME}_cuda_lib"
# ---

add_library("${PROJECT_NAME}_cuda_lib" STATIC)

set_property(TARGET "${PROJECT_NAME}_cuda_lib" PROPERTY CUDA_ARCHITECTURES OFF)

target_link_libraries("${PROJECT_NAME}_cuda_lib" PRIVATE Qt5::Core)

target_include_directories("${PROJECT_NAME}_cuda_lib" PUBLIC ${HOLO_INCLUDES})

add_subdirectory(cuda_sources) # Add cuda sources to the target



# ---
# Target "${PROJECT_NAME}_lib"
# ---

add_library("${PROJECT_NAME}_lib" STATIC
     # CMake will automatically generate C++ headers
     # (ui_xxx.h) from the following QT .ui files.
     mainwindow.ui
     plotwindow.ui
)

target_compile_features("${PROJECT_NAME}_lib" PUBLIC cxx_std_17)

# Add sources to the target and includes that
# need to be compiled to the target
add_subdirectory(sources)
add_subdirectory(includes)

if(COMMAND target_precompile_headers)
    target_precompile_headers("${PROJECT_NAME}_lib" PUBLIC "includes/stdafx.hh")
endif()

target_link_libraries("${PROJECT_NAME}_lib" PUBLIC
    Boost::boost
    Qt5::Charts
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    OpenGL::GL

    cufft
    nppc
    nppial
    nppif
    nppist
    cublas
    cusolver

    "${PROJECT_NAME}_cuda_lib"
)

# On MSVC, the Boost headers contains #pragma directives pointing to the
# corresponding libraries, automatically linking them.
if(MSVC)
    target_link_directories("${PROJECT_NAME}_lib" PUBLIC "${Boost_LIBRARY_DIRS}")
else()
    target_link_libraries("${PROJECT_NAME}_lib" PUBLIC Boost::program_options)
endif(MSVC)

# Quick fix: https://stackoverflow.com/questions/44746308/how-to-make-include-directories-added-with-autouic-available-to-downstream-targe
add_custom_command(TARGET "${PROJECT_NAME}_lib" POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}_lib_autogen/include/"
        "${CMAKE_CURRENT_BINARY_DIR}/Holovibes_autogen/include/"
)



# ---
# Targets test_* (unit tests)
# ---

set (GTEST_LIBRARY      "C:/gtest/googletest/msvc/2010/gtest-md/x64-Debug/gtestd.lib")
set (GTEST_ROOT         "C:/gtest/googletest")
set (GTEST_MAIN_LIBRARY "C:/gtest/googletest/msvc/2010/gtest-md/x64-Debug/gtestd.lib")

set(SRC_TESTS
    tests/test_example.cc
    tests/test_queue.cc
    # Add your test files here
)

enable_testing()
find_package(GTest)
find_package (Threads)
if (CMAKE_BUILD_TYPE STREQUAL "Debug" AND ${GTEST_FOUND} AND ${THREADS_FOUND})
    include_directories(${GTEST_INCLUDE_DIRS})
    message(STATUS "TESTS: Unit tests enabled")

    # For each test file create an executable of test. Launched using ctest
    foreach(f IN LISTS SRC_TESTS)
        # Find the test_name by deleting the extension and the parent_dir
        string(REGEX REPLACE "\\.[^.]*$" "" F_WITHOUT_EXT ${f})
        string(REGEX REPLACE "[^/]*/" "" TEST_NAME ${F_WITHOUT_EXT})

        # Create an executable with all sources and lib needed
        add_executable(${TEST_NAME} ${f})
        target_link_libraries(${TEST_NAME} PRIVATE
            "${PROJECT_NAME}_lib"
            ${GTEST_BOTH_LIBRARIES}
        )

        set_target_properties(${TEST_NAME} PROPERTIES
            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/")

        if(MSVC)
            target_link_directories(${TEST_NAME} PRIVATE "${Boost_LIBRARY_DIRS}")
        else()
            target_link_libraries(${TEST_NAME} PRIVATE Boost::program_options)
        endif(MSVC)

        # Add the executable created to the test list of ctest
        add_test(NAME ${TEST_NAME} COMMAND ${TEST_NAME})
        # Set timeout of the test to 5 seconds
        set_tests_properties(${TEST_NAME} PROPERTIES TIMEOUT 5)
    endforeach()
else()
    message("TESTS: Unit tests disabled in Release mode")
endif()



# ---
# Targets Holovibes
# ---

add_executable("${PROJECT_NAME}"
    mainwindow.ui
    plotwindow.ui

    Holovibes.rc

    sources/main.cc
)

target_link_libraries("${PROJECT_NAME}" PRIVATE "${PROJECT_NAME}_lib")

# Generate Holovibes.exe in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}" PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)

target_link_options("${PROJECT_NAME}" PRIVATE
    # Holovibes requires elevated privilege on Windows 10 to write
    # to *.ini files inside Program Files/
    # TODO: put *.ini files in %APPDATA%
    "$<$<NOT:$<CONFIG:Debug>>:/MANIFESTUAC:level='requireAdministrator' uiAccess='false'>"
)



# ---
# Post build: Copy necessary files to output dir
# ---

set(LIB_DIR "${CMAKE_SOURCE_DIR}/libs")
set(HOLO_DIR "${CMAKE_SOURCE_DIR}/Holovibes")
set(CAMERA_DIR "${CMAKE_SOURCE_DIR}/Camera")
set(BUILD_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")
add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD
    # Copy shaders and convolution kernels
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/shaders/" "${BUILD_DIR}/shaders/"
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${HOLO_DIR}/convolution_kernels/" "${BUILD_DIR}/convolution_kernels/"

    # Copy camera libs
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ximea/xiapi64.dll" "${BUILD_DIR}/xiapi64.dll"
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/ids/uEye_api_64.dll" "${BUILD_DIR}/uEye_api_64.dll"
    COMMAND ${CMAKE_COMMAND} -E copy "${LIB_DIR}/hamamatsu/dcamapi.dll" "${BUILD_DIR}/dcamapi.dll"

    # Copy camera .ini configs
    COMMAND ${CMAKE_COMMAND} -E copy_directory "${CAMERA_DIR}/configs/" "${BUILD_DIR}/"

    # Copy icon and logo
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/Holovibes.ico" "${BUILD_DIR}/Holovibes.ico"
    COMMAND ${CMAKE_COMMAND} -E copy "${HOLO_DIR}/holovibes_logo.png" "${BUILD_DIR}/holovibes_logo.png"
)

# Copy holovibes .ini config if it exists,
# otherwise it will be generated by Holovibes.exe on startup.
if (EXISTS "${HOLO_DIR}/holovibes.ini")
    add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${HOLO_DIR}/holovibes.ini"
            "${CMAKE_BINARY_DIR}/$<CONFIG>/holovibes.ini"
    )
endif()