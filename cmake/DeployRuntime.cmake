if(NOT DEFINED DEPLOY_DESTINATION)
    get_filename_component(DEPLOY_DESTINATION "${deploy_executable}" DIRECTORY)
endif()
file(MAKE_DIRECTORY "${DEPLOY_DESTINATION}")
# Driver DLLs are supplied by the NVIDIA driver. Windows API sets and system
# libraries must not be redistributed as application dependencies.
file(GET_RUNTIME_DEPENDENCIES
    EXECUTABLES "${deploy_executable}"
    LIBRARIES ${deploy_libraries} ${deploy_extra_files}
    DIRECTORIES ${deploy_search_dirs}
    RESOLVED_DEPENDENCIES_VAR resolved
    UNRESOLVED_DEPENDENCIES_VAR unresolved
    PRE_EXCLUDE_REGEXES
        "^api-ms-" "^ext-ms-"
        "^[Nn][Vv][Cc][Uu][Dd][Aa]\\.[Dd][Ll][Ll]$" "^[Nn][Vv][Mm][Ll]\\.[Dd][Ll][Ll]$"
        # Optional delay-loaded IDS acceleration libraries come from the IDS
        # driver installation, which is required only when using that camera.
        "^[Gg][Pp][Uu][Aa][Cc][Cc]_64\\.[Dd][Ll][Ll]$" "^[Vv][Cc][Oo][Mm][Pp]90\\.[Dd][Ll][Ll]$"
    POST_EXCLUDE_REGEXES ".*[Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\][Ss][Yy][Ss][Tt][Ee][Mm]32[/\\\\].*"
)
if(unresolved)
    message(FATAL_ERROR "Unresolved runtime dependencies: ${unresolved}")
endif()
foreach(dll IN LISTS resolved deploy_libraries deploy_extra_files)
    get_filename_component(dll_name "${dll}" NAME)
    set(destination "${DEPLOY_DESTINATION}/${dll_name}")
    if(NOT dll STREQUAL destination)
        file(COPY_FILE "${dll}" "${destination}" ONLY_IF_DIFFERENT)
    endif()
endforeach()
