project (BitflowCyton CXX CUDA)

add_library("${PROJECT_NAME}" SHARED
    bitflow_cyton.cc
    ../Camera/camera_logger.cc
)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${BITFLOW_SDK}/Include/"
    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
    "${CAMERA_DIR}/Camera/"
    "${CAMERA_DIR}/include/"
    "${HOLO_DIR}/includes/"
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
    Boost::boost
    CUDA::cudart
    spdlog::spdlog_header_only
)
