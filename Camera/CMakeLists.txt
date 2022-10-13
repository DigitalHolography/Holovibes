include(CheckIncludeFile)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Idealy create a FindBitflow.cmake file
set(BITFLOW_SDK "C:/BitFlow SDK 6.6/")
# idem
set(EURESYS_SDK "C:/Program Files/Euresys/eGrabber")

find_file(HAVE_BFType "BFType.h" PATHS "${BITFLOW_SDK}/Include")
find_file(HAVE_BFErApi "BFErApi.h" PATHS "${BITFLOW_SDK}/Include")
find_file(HAVE_BiDef "BiDef.h" PATHS "${BITFLOW_SDK}/Include")
find_file(HAVE_BiApi "BiApi.h" PATHS "${BITFLOW_SDK}/Include")

find_file(HAVE_EGrabber "EGrabber.h" PATHS "${EURESYS_SDK}/include")
find_file(HAVE_EGrabbers "EGrabbers.h" PATHS "${EURESYS_SDK}/include")

if (HAVE_BFType AND HAVE_BiDef AND HAVE_BiApi AND HAVE_BFErApi)
    add_subdirectory(BitflowCyton)
endif()

if (HAVE_BFType AND HAVE_BiDef AND HAVE_BiApi)
    add_subdirectory(CameraAdimec)
endif()

if (HAVE_EGrabber AND HAVE_EGrabbers)
    add_subdirectory(CameraPhantom)
endif()

add_subdirectory(CameraHamamatsu)
add_subdirectory(CameraIds)
add_subdirectory(CameraXib)
add_subdirectory(CameraXiq)
add_subdirectory(CameraOpenCV)
