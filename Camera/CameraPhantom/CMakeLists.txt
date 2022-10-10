project (CameraPhantom CXX CUDA)

add_library("${PROJECT_NAME}" SHARED
    camera_phantom.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${EURESYS_SDK}/include/"

    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"

    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Holovibes/includes/"
)

target_link_libraries("${PROJECT_NAME}"
    # Dirty fix to retrieve static libs of CUDA
    # "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}/../lib/x64/cudart.lib"

    Boost::boost
    CUDA::cudart
)

# Generate the dll file in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)
