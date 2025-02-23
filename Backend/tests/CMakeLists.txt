find_package(GTest REQUIRED)

include(GoogleTest)
enable_testing()

set(INTERNAL_INCLUDES_TEST
    "${HOLO_DIR}/cuda_includes/"
    "${HOLO_DIR}/cuda_includes/analysis/"
    "${HOLO_DIR}/includes/"
    "${HOLO_DIR}/includes/api/"
    "${HOLO_DIR}/includes/analysis/"
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
    "${HOLO_DIR}/includes/tools/misc/"
    "${HOLO_DIR}/includes/tools/state/"
    "${HOLO_DIR}/includes/tools/units/"
    "${CAMERA_DIR}/include/"
    "${HOLO_DIR}/tests/includes"
)

file(GLOB_RECURSE UNIT_TEST_SOURCES *.cc)

add_executable(test
    ${UNIT_TEST_SOURCES}
)

target_include_directories(test PRIVATE
    ${INTERNAL_INCLUDES_TEST}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${nlohmann_json_INCLUDE_DIRS}
)

target_link_libraries(test
    GTest::gtest_main
    spdlog::spdlog_header_only
    Backend
)

target_compile_options(test PRIVATE
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

gtest_discover_tests(test)