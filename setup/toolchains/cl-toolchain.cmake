cmake_policy(SET CMP0091 NEW)
set(CMAKE_C_COMPILER cl)
set(CMAKE_CXX_COMPILER cl)

set(CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_LINKER "link.exe")

#Remove 'no debug *.pdg file found' warning cause conan dont keep them
string(APPEND CMAKE_EXE_LINKER_FLAGS " /ignore:4099")

# "/external:anglebrackets" : Tells MSVC that all #include <path/to/.h> are from external sources
# "/external:W0"            : Tells MSVC that external headers should not do any warnings
string(APPEND CMAKE_CXX_FLAGS " /external:anglebrackets /external:W0")

add_definitions(-DUNICODE -D_UNICODE)

string(APPEND CMAKE_EXE_LINKER_FLAGS " /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBCMTD")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CXX_FLAGS " /Od /DEBUG /Zi /EHa")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /NODEFAULTLIB:MSVCRT /DEFAULTLIB:MSVCRTD")
else()
    string(APPEND CMAKE_CXX_FLAGS " /O2 /EHsc ")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /NODEFAULTLIB:MSVCRTD /DEFAULTLIB:MSVCRT")
endif()
