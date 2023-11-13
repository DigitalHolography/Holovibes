set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)

set(CXX_STANDARD 20)

add_definitions(-DUNICODE -D_UNICODE)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    string(APPEND CMAKE_CXX_FLAGS " -g3 -O0")
else()
    string(APPEND CMAKE_CXX_FLAGS " -O3")
endif()
