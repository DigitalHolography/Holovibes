cmake_minimum_required (VERSION 3.0)

project (CameraAdimec CXX)

find_package(Boost REQUIRED)

add_library("${PROJECT_NAME}" SHARED
    camera_adimec.cc
)

set(BITFLOW_SDK "C:/BitFlow SDK 6.30/")

include_directories(
    "${BITFLOW_SDK}/Include/"
    "${Boost_INCLUDE_DIRS}"

    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/Camera/include/"
)

target_link_libraries("${PROJECT_NAME}"
    "${BITFLOW_SDK}/Lib64/BFD.lib"
    "${BITFLOW_SDK}/Lib64/BFDiskIO.lib"
    "${BITFLOW_SDK}/Lib64/BFEr.lib"
    "${BITFLOW_SDK}/Lib64/BFOS.lib"
    "${BITFLOW_SDK}/Lib64/BFS.lib"
    "${BITFLOW_SDK}/Lib64/Bid.lib"
    "${BITFLOW_SDK}/Lib64/BufferInterface.lib"
    "${BITFLOW_SDK}/Lib64/Cid.lib"
    "${BITFLOW_SDK}/Lib64/DispSurf.lib"
    "${BITFLOW_SDK}/Lib64/Gn2d.lib"
    "${BITFLOW_SDK}/Lib64/R2d.lib"
    "${BITFLOW_SDK}/Lib64/R64d.lib"
    "${BITFLOW_SDK}/Lib64/clallserial.lib"
    "${BITFLOW_SDK}/Lib64/clserbit.lib"
)
