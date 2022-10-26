cmake_policy(SET CMP0091 NEW)
set(CMAKE_C_COMPILER clang-cl)
set(CMAKE_CXX_COMPILER clang-cl)
set(CMAKE_EXE_LINKER_FLAGS    "${CMAKE_EXE_LINKER_FLAGS} /MANIFEST:NO")
set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} /MANIFEST:NO")
set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} /MANIFEST:NO")

set(CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)
set(MSVC_INCREMENTAL_DEFAULT ON)

# Default linker is needed because some conan packages are breaking lld-link
set(CMAKE_LINKER "link.exe")

string(APPEND CMAKE_CXX_FLAGS " /std:c++20")

#Remove 'no debug *.pdg file found' warning cause conan dont keep them
string(APPEND CMAKE_EXE_LINKER_FLAGS " /ignore:4099")

#add_compile_options(-fuse-ld=lld-link)
set(MSVC_INCREMENTAL_DEFAULT ON)

add_definitions(-DWIN32 -DNOMINMAX -DWIN32_LEAN_AND_MEAN -D_XKEYCHECK_H -D_CRT_SECURE_NO_WARNINGS -D_CRT_NONSTDC_NO_WARNINGS -D_WIN32_WINNT=0x0600 -DUNICODE -D_UNICODE -DFMT_HEADER_ONLY )
string(APPEND CMAKE_CXX_FLAGS " -fms-extensions -fms-compatibility -Wno-ignored-attributes -Wno-unused-local-typedef -Wno-expansion-to-defined -Wno-pragma-pack -Wno-ignored-pragma-intrinsic -Wno-unknown-pragmas -Wno-invalid-token-paste -Wno-deprecated-declarations -Wno-macro-redefined -Wno-dllimport-static-field-def -Wno-unused-command-line-argument -Wno-unknown-argument -Wno-int-to-void-pointer-cast")

#Make find boost happy
#set(MSVC_TOOLSET_VERSION 144)

string(APPEND CMAKE_EXE_LINKER_FLAGS " /NODEFAULTLIB:LIBCMT /NODEFAULTLIB:LIBCMTD")
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "DEBUG BUILD")
    string(APPEND CMAKE_CXX_FLAGS " /Od /DEBUG /Z7 /EHa /Ob0")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /NODEFAULTLIB:MSVCRT /DEFAULTLIB:MSVCRTD")
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_TRACE)
else()
    message(STATUS "RELEASE BUILD")
    string(APPEND CMAKE_CXX_FLAGS " /O2 /EHsc ")
    string(APPEND CMAKE_EXE_LINKER_FLAGS " /NODEFAULTLIB:MSVCRTD /DEFAULTLIB:MSVCRT")
    add_definitions(-DSPDLOG_ACTIVE_LEVEL=SPDLOG_LEVEL_INFO)
endif()
