cmake_minimum_required(VERSION 3.0)

project(Holovibes CXX CUDA)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)

set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 COMPONENTS Widgets Core Charts REQUIRED)
find_package(OpenGL REQUIRED)
find_package(Boost REQUIRED
    COMPONENTS
    program_options
)
link_directories("${Boost_LIBRARY_DIRS}")

add_executable("${PROJECT_NAME}"
    mainwindow.ui
    plotwindow.ui
)

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