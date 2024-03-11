project (gpib CXX)

add_library("${PROJECT_NAME}" SHARED
    gpib_controller.cc
)

target_compile_features("${PROJECT_NAME}" PUBLIC cxx_std_20)

target_include_directories("${PROJECT_NAME}" PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}/"
    "${CMAKE_SOURCE_DIR}/libs/ni_gpib/include/"
)

target_link_libraries("${PROJECT_NAME}"
    "${CMAKE_SOURCE_DIR}/libs/ni_gpib/nivisa64.lib"
    Boost::boost
    spdlog::spdlog_header_only
)
