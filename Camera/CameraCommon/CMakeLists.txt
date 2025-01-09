project(CameraCommon CXX CUDA)

# Crée une bibliothèque statique ou dynamique contenant les fonctionnalités communes
add_library(CameraCommon STATIC
    ../Camera/camera_logger.cc
)

# Inclure les dépendances nécessaires
target_include_directories(CameraCommon PUBLIC
    "${CMAKE_SOURCE_DIR}/Camera/Camera/"
    "${CMAKE_SOURCE_DIR}/Camera/include/"
    "${CMAKE_SOURCE_DIR}/Holovibes/includes/"
)

# Lier les dépendances
target_link_libraries(CameraCommon PUBLIC
    Boost::boost
    CUDA::cudart
    spdlog::spdlog_header_only
)