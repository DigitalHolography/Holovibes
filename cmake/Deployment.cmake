# Deploy imported DLLs explicitly, including CUDA and dynamically loaded cameras.
# vcpkg handles its own app-local dependencies; this also covers external SDKs.
function(holovibes_deploy_runtime target)
    get_filename_component(compiler_dir "${CMAKE_CXX_COMPILER}" DIRECTORY)
    find_program(HOLOVIBES_DUMPBIN dumpbin HINTS "${compiler_dir}" REQUIRED)
    set(runtime_libraries)
    foreach(camera IN LISTS ARGN)
        list(APPEND runtime_libraries "$<TARGET_FILE:${camera}>")
    endforeach()
    get_property(camera_files GLOBAL PROPERTY HOLOVIBES_CAMERA_RUNTIME_FILES)
    list(REMOVE_DUPLICATES camera_files)
    set(search_dirs
    "$<TARGET_FILE_DIR:${target}>"
    "${VCPKG_INSTALLED_DIR}/${VCPKG_TARGET_TRIPLET}/$<$<CONFIG:Debug>:debug/>bin"
    "$ENV{SystemRoot}/System32"
    "${CUDAToolkit_BIN_DIR}"
    "${CUDAToolkit_BIN_DIR}/x64"
)
    foreach(camera_file IN LISTS camera_files)
        get_filename_component(camera_dir "${camera_file}" DIRECTORY)
        list(APPEND search_dirs "${camera_dir}")
    endforeach()
    list(REMOVE_DUPLICATES search_dirs)
    set(script "${CMAKE_CURRENT_BINARY_DIR}/deploy-${target}-$<CONFIG>.cmake")
    file(GENERATE OUTPUT "${script}" CONTENT
"set(deploy_executable [==[$<TARGET_FILE:${target}>]==])
set(deploy_libraries [==[${runtime_libraries}]==])
set(deploy_extra_files [==[${camera_files}]==])
set(deploy_search_dirs [==[${search_dirs}]==])
set(CMAKE_GET_RUNTIME_DEPENDENCIES_PLATFORM windows+pe)
set(CMAKE_GET_RUNTIME_DEPENDENCIES_TOOL dumpbin)
set(CMAKE_GET_RUNTIME_DEPENDENCIES_COMMAND [==[${HOLOVIBES_DUMPBIN}]==])
include([==[${PROJECT_SOURCE_DIR}/cmake/DeployRuntime.cmake]==])
")
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND "${CMAKE_COMMAND}" -P "${script}"
        COMMENT "Deploying runtime dependencies for ${target}"
        VERBATIM
    )
    set_property(TARGET ${target} APPEND PROPERTY LINK_DEPENDS
        "${PROJECT_SOURCE_DIR}/cmake/DeployRuntime.cmake" "${script}")
    if(target STREQUAL "Holovibes")
        # The generated script is reused with an installation destination.
        install(CODE "set(DEPLOY_DESTINATION \"\$ENV{DESTDIR}\${CMAKE_INSTALL_PREFIX}\")\ninclude(\"${script}\")" COMPONENT application)
    endif()
endfunction()
