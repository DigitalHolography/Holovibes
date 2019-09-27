project (CameraPhotoFocus CXX)

add_library("${PROJECT_NAME}" SHARED
    camera_photon_focus.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/lib/photonfocus/Includes/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/lib/photonfocus/PvDevice64.lib"
    "${CMAKE_SOURCE_DIR}/lib/photonfocus/PvBase64.lib"
    "${CMAKE_SOURCE_DIR}/lib/photonfocus/PvGenICam64.lib"
    "${CMAKE_SOURCE_DIR}/lib/photonfocus/PvBuffer64.lib"
    "${CMAKE_SOURCE_DIR}/lib/photonfocus/PvStreamRaw64.lib"
    "${CMAKE_SOURCE_DIR}/lib/photonfocus/PvStream64.lib"

    Boost::boost
)

# Generate the dll file in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)