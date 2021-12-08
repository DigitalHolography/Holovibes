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

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CXX_FLAGS " /Od /MDd /DEBUG /Zi /EHa")
else()
    string(APPEND CMAKE_CXX_FLAGS " /O2 /EHsc")
endif()