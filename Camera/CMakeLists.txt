# SDK locations may be overridden in CMakeUserPresets.json or with -D.
set(BITFLOW_SDK "C:/BitFlow SDK 6.6" CACHE PATH "BitFlow SDK root")
set(PCO_SDK "C:/Program Files/PCO Digital Camera Toolbox/pco.cpp" CACHE PATH "PCO C++ SDK root")
set(EURESYS_SDK "C:/Program Files/Euresys/eGrabber" CACHE PATH "Euresys eGrabber root")
set(VIMBAX_SDK "C:/Program Files/Allied Vision/Vimba X" CACHE PATH "Vimba X root")
set(ASI_SDK "C:/Program Files/ASI SDK" CACHE PATH "ASI SDK root")

function(holovibes_collect_camera_targets directory)
    get_property(targets DIRECTORY "${directory}" PROPERTY BUILDSYSTEM_TARGETS)
    foreach(target IN LISTS targets)
        get_target_property(type ${target} TYPE)
        if(type STREQUAL "SHARED_LIBRARY")
            target_link_libraries(${target} holovibes_headers)
            set_property(GLOBAL APPEND PROPERTY HOLOVIBES_CAMERA_TARGETS ${target})
        endif()
    endforeach()
    get_property(children DIRECTORY "${directory}" PROPERTY SUBDIRECTORIES)
    foreach(child IN LISTS children)
        holovibes_collect_camera_targets("${child}")
    endforeach()
endfunction()

# AUTO retains the release build's existing SDK auto-detection. ON fails early
# when an explicitly requested SDK is missing; OFF skips the plugin.
function(holovibes_camera name directory available)
    set(option_name "HOLOVIBES_CAMERA_${name}")
    set(${option_name} AUTO CACHE STRING "Build ${name} camera: AUTO, ON or OFF")
    set_property(CACHE ${option_name} PROPERTY STRINGS AUTO ON OFF)
    if(NOT "${${option_name}}" MATCHES "^(AUTO|ON|OFF)$")
        message(FATAL_ERROR "${option_name} must be AUTO, ON or OFF")
    endif()
    if(${option_name} STREQUAL "ON" AND NOT available)
        message(FATAL_ERROR "${name} SDK is missing. Set its SDK root or disable ${option_name}.")
    endif()
    if(NOT ${option_name} STREQUAL "OFF" AND available)
        add_subdirectory(${directory})
        holovibes_collect_camera_targets("${CMAKE_CURRENT_SOURCE_DIR}/${directory}")
        set_property(GLOBAL APPEND PROPERTY HOLOVIBES_CAMERA_RUNTIME_FILES ${ARGN})
        message(STATUS "Camera ${name}: enabled")
    else()
        message(STATUS "Camera ${name}: disabled")
    endif()
endfunction()

holovibes_camera(HAMAMATSU CameraHamamatsu TRUE "${CAMERA_LIB_DIR}/hamamatsu/dcamapi.dll")
holovibes_camera(IDS CameraIds TRUE "${CAMERA_LIB_DIR}/ids/uEye_api_64.dll")
holovibes_camera(XIQ CameraXiq TRUE "${CAMERA_LIB_DIR}/ximea/xiapi64.dll")
holovibes_camera(XIB CameraXib TRUE "${CAMERA_LIB_DIR}/ximea/xiapi64.dll")
holovibes_camera(OPENCV CameraOpenCV TRUE)

set(bitflow_available TRUE)
foreach(header BFType.h BFErApi.h BiDef.h BiApi.h)
    if(NOT EXISTS "${BITFLOW_SDK}/Include/${header}")
        set(bitflow_available FALSE)
    endif()
endforeach()
holovibes_camera(BITFLOW BitflowCyton "${bitflow_available}")
holovibes_camera(ADIMEC CameraAdimec "${bitflow_available}")

set(phantom_available FALSE)
if(EXISTS "${EURESYS_SDK}/include/EGrabber.h" AND EXISTS "${EURESYS_SDK}/include/EGrabbers.h")
    set(phantom_available TRUE)
endif()
holovibes_camera(PHANTOM CamerasPhantom "${phantom_available}")

set(pco_available FALSE)
if(EXISTS "${PCO_SDK}/pco.camera/camera.h")
    set(pco_available TRUE)
endif()
holovibes_camera(PCO CameraPCO "${pco_available}")

set(asi_available FALSE)
if(EXISTS "${ASI_SDK}/include/ASICamera2.h")
    set(asi_available TRUE)
endif()
holovibes_camera(ASI CameraASI "${asi_available}" "${ASI_SDK}/lib/x64/ASICamera2.dll")

set(vimbax_available FALSE)
if(EXISTS "${VIMBAX_SDK}/api/include/VmbCPP/VmbCPP.h")
    set(vimbax_available TRUE)
endif()
# GenICam plugins can be loaded dynamically, so preserve the explicit runtime list.
holovibes_camera(ALVIUM CameraAlvium "${vimbax_available}"
    "${VIMBAX_SDK}/bin/VmbCPP.dll"
    "${VIMBAX_SDK}/bin/VmbC.dll"
    "${VIMBAX_SDK}/bin/GenApi_MD_VC141_v3_2_AVT.dll"
    "${VIMBAX_SDK}/bin/GCBase_MD_VC141_v3_2_AVT.dll"
    "${VIMBAX_SDK}/bin/XmlParser_MD_VC141_v3_2_AVT.dll"
    "${VIMBAX_SDK}/bin/Log_MD_VC141_v3_2_AVT.dll"
    "${VIMBAX_SDK}/bin/NodeMapData_MD_VC141_v3_2_AVT.dll"
    "${VIMBAX_SDK}/bin/MathParser_MD_VC141_v3_2_AVT.dll"
)
