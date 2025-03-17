include(CheckIncludeFile)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Idealy create a FindBitflow.cmake file
set(BITFLOW_SDK "C:/BitFlow SDK 6.6/")

set(EURESYS_SDK "C:/Program Files/Euresys/eGrabber")
set(VIMBAX_SDK "C:/Program Files/Allied Vision/Vimba X")
set(ASI_SDK "C:/Program Files/ASI SDK")

find_file(HAVE_BFType "BFType.h" PATHS "${BITFLOW_SDK}/Include")
find_file(HAVE_BFErApi "BFErApi.h" PATHS "${BITFLOW_SDK}/Include")
find_file(HAVE_BiDef "BiDef.h" PATHS "${BITFLOW_SDK}/Include")
find_file(HAVE_BiApi "BiApi.h" PATHS "${BITFLOW_SDK}/Include")


find_file(HAVE_EGrabber "EGrabber.h" PATHS "${EURESYS_SDK}/include")
find_file(HAVE_EGrabbers "EGrabbers.h" PATHS "${EURESYS_SDK}/include")

find_file(HAVE_VIMBAX "VmbCPP.h" PATHS "${VIMBAX_SDK}/api/include/VmbCPP")

find_file(HAVE_ASI "ASICamera2.h" PATHS "${ASI_SDK}/include")

if (HAVE_BFType AND HAVE_BiDef AND HAVE_BiApi AND HAVE_BFErApi)
    add_subdirectory(BitflowCyton)
else()
    message(WARNING "#####################################################\nBitflow SDK not found, BitflowCyton will not be built\n#####################################################")
endif()

if (HAVE_BFType AND HAVE_BiDef AND HAVE_BiApi)
    add_subdirectory(CameraAdimec)
else()
    message(WARNING "#####################################################\nBitflow SDK not found, CameraAdimec will not be built\n#####################################################")
endif()

if (HAVE_EGrabber AND HAVE_EGrabbers)
    add_subdirectory(CamerasPhantom)
else()
    message(WARNING "######################################################\nEuresys SDK not found, CameraPhantom will not be built\n######################################################")
endif()

if (HAVE_VIMBAX)
    add_subdirectory(CameraAlvium)

else()
    message(WARNING "######################################################\nVimba X SDK not found, Camera Alvium will not be built\n######################################################")
endif()

if (HAVE_ASI)
    add_subdirectory(CameraASI)

else()
    message(WARNING "######################################################\nASI SDK not found, Camera ASI will not be built\n######################################################")
endif()

add_subdirectory(CameraHamamatsu)
add_subdirectory(CameraIds)
add_subdirectory(CameraXib)
add_subdirectory(CameraXiq)
add_subdirectory(CameraOpenCV)
