project(Holovibes CXX CUDA)

# Qt file generators
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOUIC_SEARCH_PATHS "${CMAKE_CURRENT_SOURCE_DIR}/")

# Packages
find_package(Qt5 COMPONENTS Widgets Core Charts REQUIRED)
find_package(OpenGL REQUIRED)

# Add Qt ui files
add_executable("${PROJECT_NAME}"
    mainwindow.ui
    plotwindow.ui
    Holovibes.rc
)

# C++ standard
target_compile_features("${PROJECT_NAME}" PRIVATE cxx_std_17)

target_link_options("${PROJECT_NAME}"
    PRIVATE

    # Holovibes requires elevated privilege on Windows 10 to write
    # to *.ini files inside Program Files/
    # TODO: put *.ini files in %APPDATA%
    "$<$<NOT:$<CONFIG:Debug>>:/MANIFESTUAC:level='requireAdministrator' uiAccess='false'>"
)

add_library(holocuda STATIC)

target_include_directories(holocuda
    PUBLIC
    cuda_includes/
    includes/
    includes/compute/
    includes/thread/
    includes/tools/
    includes/tools/cuda_tools/
    includes/tools/units/

    "${CMAKE_SOURCE_DIR}/Camera/include/"
)

target_link_libraries(holocuda
    Qt5::Core
)

# On msvc, the Boost headers contains #pragma directives pointing to the
# corresponding libraries, automatically linking them.
if(MSVC)
    target_link_directories("${PROJECT_NAME}"
        PRIVATE
        "${Boost_LIBRARY_DIRS}"
    )
else()
    target_link_libraries("${PROJECT_NAME}"
        Boost::program_options
    )
endif(MSVC)

# Add necessary include dirs
target_include_directories("${PROJECT_NAME}"
    PRIVATE
    includes/core/
    includes/gui/other/
    includes/gui/selections/
    includes/gui/selections/interfaces/
    includes/gui/selections/overlays/
    includes/gui/windows/
    includes/thread/

    "${CMAKE_SOURCE_DIR}/lib/glm/"
    "${CMAKE_SOURCE_DIR}/GPIB/"

    "${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}"
)

# CMake supports precompiled headers natively starting with v3.16
if(COMMAND target_precompile_headers)
    target_precompile_headers("${PROJECT_NAME}"
        PUBLIC
        "includes/stdafx.hh"
    )
endif()

add_subdirectory(sources)
add_subdirectory(cuda_sources)
add_subdirectory(includes)

target_link_libraries("${PROJECT_NAME}"
    Boost::boost
    Qt5::Charts
    Qt5::Core
    Qt5::Gui
    Qt5::Widgets
    OpenGL::GL

    cufft
    nppc
    nppial
    nppif
    nppist

    holocuda
)

# Generate Holovibes.exe in build/Debug or build/Release
set_target_properties("${PROJECT_NAME}"
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/$<CONFIG>/"
)

# Copy necessary files to output dir
set(LIB_DIR "${CMAKE_SOURCE_DIR}/lib")
set(HOLO_DIR "${CMAKE_SOURCE_DIR}/Holovibes")
add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD
    # Copy shaders
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${HOLO_DIR}/shaders/"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/shaders/"

    # Copy convolution kernels
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${CMAKE_SOURCE_DIR}/ConvolutionKernels/"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/ConvolutionKernels/"

    # Copy camera libs
    COMMAND ${CMAKE_COMMAND} -E copy
        "${LIB_DIR}/pco/SC2_Cam.dll"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/SC2_Cam.dll"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${LIB_DIR}/ixon/atmcd64d.dll"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/atmcd64d.dll"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${LIB_DIR}/avt/FGCamera.dll"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/FGCamera.dll"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${LIB_DIR}/ximea/m3apiX64.dll"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/m3apiX64.dll"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${LIB_DIR}/ximea/xiapi64.dll"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/xiapi64.dll"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${LIB_DIR}/ids/uEye_api_64.dll"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/uEye_api_64.dll"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${LIB_DIR}/hamamatsu/dcamapi.dll"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/dcamapi.dll"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${LIB_DIR}/pixelink/lib/x64/PxLAPI40.lib"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/PxLAPI40.lib"

    # Copy camera .ini configs
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/adimec.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/adimec.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/ids.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/ids.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/edge.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/edge.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/pixelfly.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/pixelfly.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/xiq.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/xiq.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/xib.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/xib.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/pike.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/pike.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/pixelink.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/pixelink.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/ixon.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/ixon.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/photonfocus.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/photonfocus.ini"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/hamamatsu.ini"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/hamamatsu.ini"

    # Copy icon and logo
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/Holovibes.ico"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/Holovibes.ico"
    COMMAND ${CMAKE_COMMAND} -E copy
        "${HOLO_DIR}/holovibes_logo.png"
        "${CMAKE_BINARY_DIR}/$<CONFIG>/holovibes_logo.png"
)

# Copy holovibes .ini config if it exists else it will be generated by holovibes.exe on startup
if (EXISTS "${HOLO_DIR}/holovibes.ini")
    add_custom_command(TARGET "${PROJECT_NAME}" POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            "${HOLO_DIR}/holovibes.ini"
            "${CMAKE_BINARY_DIR}/$<CONFIG>/holovibes.ini"
    )
endif()
