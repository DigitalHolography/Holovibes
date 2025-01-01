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
	"${HOLO_DIR}/includes/tools/units/"
	"${HOLO_DIR}/includes/tools/check/"
	"${HOLO_DIR}/includes/tools/exception/"
	"${HOLO_DIR}/includes/tools/logger/"
	"${HOLO_DIR}/includes/tools/state/"
    "${CAMERA_DIR}/include/"
)

set(EXTERNAL_INCLUDES_TEST
    ${Boost_INCLUDE_DIRS}
    ${CUDA_EXTERNAL_INCLUDES}
)

file(GLOB_RECURSE SOURCES_TEST sources/*.cc)
file(GLOB_RECURSE HEADERS_TEST includes/*.hh)

# Exclude gui and entry point
list(FILTER SOURCES_TEST EXCLUDE REGEX "/gui/")
list(FILTER HEADERS_TEST EXCLUDE REGEX "/gui/")
list(FILTER SOURCES_TEST EXCLUDE REGEX "main.cc")

add_executable(test
	${SOURCES_TEST}
    ${HEADERS_TEST}
    ${CUDA_SOURCES}
    ${CUDA_HEADERS}
	test_example.cc
	test_chart.cc
)

target_include_directories(test PRIVATE
    ${EXTERNAL_INCLUDES_TEST}
    ${INTERNAL_INCLUDES_TEST}
)

target_link_libraries(test
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
)

gtest_discover_tests(test)