cmake_minimum_required(VERSION 3.24)

project(Holovibes
	VERSION 14.4.2
	DESCRIPTION "Holovibes"
	LANGUAGES CXX CUDA
)

# Export compile_commands.json so IDE can use it for code completion.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

set(HOLO_DIR "${CMAKE_SOURCE_DIR}/Backend")
set(UI_DIR "${CMAKE_SOURCE_DIR}/Ui")
set(CAMERA_DIR "${CMAKE_SOURCE_DIR}/Camera")
set(CAMERA_LIB_DIR "${CMAKE_SOURCE_DIR}/Camera/libs")
set(PRESET_DIR "${UI_DIR}/assets/preset")

# Common libraries for all targets
find_package(CUDA 12.2 REQUIRED)
find_package(CUDAToolkit 12.2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(Boost COMPONENTS program_options REQUIRED)
find_package(nlohmann_json REQUIRED)
find_package(Threads)

# Setup individual targets
add_subdirectory(Camera)
add_subdirectory(Backend)
add_subdirectory(Ui)
add_subdirectory(docs)
