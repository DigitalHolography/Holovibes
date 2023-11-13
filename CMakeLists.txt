cmake_minimum_required(VERSION 3.24)

project(Holovibes)

# Create all the FindXXX.cmake needed for the project
# set(CONAN_CMAKE_SILENT_OUTPUT ON)
# include(${CMAKE_BINARY_DIR}/conan_paths.cmake)

set(LIB_DIR "${CMAKE_SOURCE_DIR}/libs")
set(HOLO_DIR "${CMAKE_SOURCE_DIR}/Holovibes")
set(CAMERA_DIR "${CMAKE_SOURCE_DIR}/Camera")
set(BUILD_DIR "${CMAKE_BINARY_DIR}/$<CONFIG>")
# set(SPDLOG_DIR "${CONAN_SPDLOG_ROOT}/")
# set(OPENCV_DIR "${CONAN_OPENCV_ROOT}/")
set(EURESYS_DIR "${CMAKE_SOURCE_DIR}/setup_710")

# Used in every sub project
find_package(Boost REQUIRED
	COMPONENTS
	program_options
)
find_package(OpenCV REQUIRED)

# Add spdlog to include path || don't move it bellow
find_package(CUDAToolkit 11.5 REQUIRED)
find_package(spdlog REQUIRED)
# include_directories("${SPDLOG_DIR}/include")


set(EXTERNAL_INCLUDES
    ${opencv_INCLUDE_DIRS}
)

add_subdirectory(Camera)
add_subdirectory(GPIB)
add_subdirectory(Holovibes)
# add_subdirectory(docs)
