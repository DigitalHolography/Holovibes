set(EXTERNAL_INCLUDES
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${nlohmann_json_INCLUDE_DIRS}
)

set(INTERNAL_INCLUDES
    cuda_includes/
    includes/
    includes/api/
    includes/compute/
    includes/data_struct/
    includes/core/
    includes/enum/
    includes/struct/
    includes/io/
    includes/io/input_file/
    includes/io/output_file/
    includes/thread/
    includes/tools/
    includes/tools/cuda_tools/
    includes/tools/units/
    includes/tools/check/
    includes/tools/exception/
    includes/tools/logger/
    includes/tools/state/
    "${CAMERA_DIR}/include/"
)

# ---------------------------------------------------------------------
# The executable
# ---------------------------------------------------------------------
file(GLOB_RECURSE SOURCES sources/*.cc)
file(GLOB_RECURSE HEADERS includes/*.hh)
file(GLOB_RECURSE CUDA_SOURCES cuda_sources/*.cu)
file(GLOB_RECURSE CUDA_HEADERS cuda_includes/*.cuh)

add_library(Backend STATIC
    ${SOURCES}
    ${HEADERS}
    ${CUDA_SOURCES}
    ${CUDA_HEADERS}
)

target_link_libraries(Backend PRIVATE
    spdlog::spdlog_header_only
    opencv_videoio
    cufft
    cublas
    cusolver
    nvml
)

target_include_directories(Backend PRIVATE
    ${EXTERNAL_INCLUDES}
    ${INTERNAL_INCLUDES}
)

set_target_properties(Backend PROPERTIES
    CUDA_ARCHITECTURES "61;75;86;89"
    CUDA_STANDARD 17
)

target_compile_options(Backend PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:
    --extended-lambda
    -Xcudafe --diag_suppress=field_without_dll_interface
    -diag-suppress 27
    >
    $<$<COMPILE_LANGUAGE:CXX>:
    /W3
    /wd4005
    /wd4003
    >
)

target_compile_definitions(Backend PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:.
    _CRT_SECURE_NO_WARNINGS
    >
)

if(COMMAND target_precompile_headers)
    target_precompile_headers(Backend PUBLIC "includes/stdafx.hh")
endif()

# ---------------------------------------------------------------------
# Unit testing
# ---------------------------------------------------------------------
add_subdirectory(tests)
