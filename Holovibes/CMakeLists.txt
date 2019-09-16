cmake_minimum_required(VERSION 3.0)

project(Holovibes CXX CUDA)

# Qt file generators
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/")

# C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Packages
find_package(Qt5 COMPONENTS Widgets Core Charts REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost REQUIRED
    COMPONENTS
    program_options
)
# Should be happening in target_link_libraries but doesn't work for some reason
link_directories("${Boost_LIBRARY_DIRS}")

# Add Qt ui files
add_executable("${PROJECT_NAME}"
    mainwindow.ui
    plotwindow.ui
)

# Add necessary include dirs
target_include_directories("${PROJECT_NAME}"
    PRIVATE
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    includes
    includes/compute/
    includes/core/
    includes/gui/other/
    includes/gui/selections/
    includes/gui/selections/interfaces/
    includes/gui/selections/overlays/
    includes/gui/windows/
    includes/thread/
    includes/tools/
    includes/tools/cuda_tools/
    includes/tools/units/
    cuda_includes

    "${Boost_INCLUDE_DIRS}"

    "${CMAKE_SOURCE_DIR}/lib/glm/"
    "${CMAKE_SOURCE_DIR}/GPIB/"
)

# Add precompiled header
target_precompile_headers("${PROJECT_NAME}"
    PUBLIC
    "stdafx.hh"
)

add_subdirectory(sources)
add_subdirectory(cuda_sources)
add_subdirectory(includes)

target_link_libraries("${PROJECT_NAME}"
    Qt5::Charts
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    cufft
    OpenGL::GL
)

# Copy shaders to the .exe dir
add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_CURRENT_SOURCE_DIR}/shaders/"
        "${CMAKE_CURRENT_BINARY_DIR}/$<CONFIG>/shaders/"
)