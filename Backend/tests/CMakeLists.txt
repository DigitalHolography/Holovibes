find_package(GTest CONFIG REQUIRED)

include(GoogleTest)

set(INTERNAL_INCLUDES_TEST
    "${HOLO_DIR}/cuda_includes/"
    "${HOLO_DIR}/includes/"
    "${HOLO_DIR}/includes/api/"
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
    "${HOLO_DIR}/tests/includes"
)

file(GLOB_RECURSE UNIT_TEST_SOURCES CONFIGURE_DEPENDS *.cc)

add_executable(HolovibesTests
    ${UNIT_TEST_SOURCES}
)

target_include_directories(HolovibesTests PRIVATE
    ${INTERNAL_INCLUDES_TEST}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    ${nlohmann_json_INCLUDE_DIRS}
)

target_link_libraries(HolovibesTests
    GTest::gtest_main
    holovibes_headers
    CUDA::cudart
    Backend
)

target_compile_options(HolovibesTests PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
    /W3
    /wd4005
    /wd4003
    >
)

target_compile_definitions(HolovibesTests PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:
    _CRT_SECURE_NO_WARNINGS
    >
)

holovibes_deploy_runtime(HolovibesTests)
gtest_discover_tests(HolovibesTests DISCOVERY_MODE PRE_TEST)