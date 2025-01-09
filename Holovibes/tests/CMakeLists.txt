include(GoogleTest)

enable_testing()

set(INTERNAL_INCLUDES_TEST
    "${HOLO_DIR}/cuda_includes/"
    "${HOLO_DIR}/includes/"
    "${HOLO_DIR}/includes/api/"
    "${HOLO_DIR}/includes/cli/"
    "${HOLO_DIR}/includes/compute/"
    "${HOLO_DIR}/includes/data_struct/"
    "${HOLO_DIR}/includes/core/"
    "${HOLO_DIR}/includes/enum/"
    "${HOLO_DIR}/includes/struct/"
    "${HOLO_DIR}/includes/io/"
    "${HOLO_DIR}/includes/io/input_file/"
    "${HOLO_DIR}/includes/io/output_file/"
    "${HOLO_DIR}/includes/thread/"
    "${HOLO_DIR}/includes/tools/"
    "${HOLO_DIR}/includes/tools/cuda_tools/"
    "${HOLO_DIR}/includes/tools/exception/"
    "${HOLO_DIR}/includes/tools/logger/"
    "${HOLO_DIR}/includes/tools/state/"
    "${HOLO_DIR}/includes/tools/units/"
    "${CAMERA_DIR}/include/"
)

set(UNIT_TEST_SOURCES
    test_example.cc
    test_batch_input_queue.cc
    test_chart.cc
    test_fast_updates_holder.cc
    test_map.cc
    test_queue.cc
    test_reduce.cc
    test_subsample_frame.cc
)

set(EXTERNAL_INCLUDES_TEST
    ${Boost_INCLUDE_DIRS}
    ${CUDA_EXTERNAL_INCLUDES}
)

file(GLOB_RECURSE SOURCES_TEST "${HOLO_DIR}/sources/*.cc")
file(GLOB_RECURSE HEADERS_TEST "${HOLO_DIR}/includes/*.hh")

# Exclude gui and entry point
list(FILTER SOURCES_TEST EXCLUDE REGEX "/gui/")
list(FILTER HEADERS_TEST EXCLUDE REGEX "/gui/")
list(FILTER SOURCES_TEST EXCLUDE REGEX "main.cc")

add_executable(test
    ${SOURCES_TEST}
    ${HEADERS_TEST}
    ${CUDA_SOURCES}
    ${CUDA_HEADERS}
    ${UNIT_TEST_SOURCES}
)

target_include_directories(test PRIVATE
    ${EXTERNAL_INCLUDES_TEST}
    ${INTERNAL_INCLUDES_TEST}
)

add_custom_command(
    TARGET test PRE_LINK
    COMMAND ${CMAKE_COMMAND} -E remove -f $<TARGET_FILE:test>
    COMMAND taskkill /F /IM test.exe || exit 0
)

target_link_libraries(test PRIVATE
    GTest::gtest_main
    spdlog::spdlog_header_only
    fmt::fmt
    Boost::program_options
    opencv_videoio
    cufft
    cublas
    cusolver
    nvml
)

target_compile_options(test PRIVATE
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

target_compile_definitions(test PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:.
    _CRT_SECURE_NO_WARNINGS
    >
)

set_target_properties(test PROPERTIES
    CUDA_ARCHITECTURES "61;75;86;89"
    CUDA_STANDARD 17
    LINK_FLAGS "/INCREMENTAL:NO"
)

gtest_discover_tests(test)