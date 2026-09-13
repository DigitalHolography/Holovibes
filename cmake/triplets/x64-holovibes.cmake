set(VCPKG_TARGET_ARCHITECTURE x64)
set(VCPKG_CRT_LINKAGE dynamic)
set(VCPKG_LIBRARY_LINKAGE dynamic)
set(VCPKG_PLATFORM_TOOLSET v143)

file(READ "${CMAKE_CURRENT_LIST_DIR}/../toolchain.json" holovibes_toolchain)
string(JSON VCPKG_PLATFORM_TOOLSET_VERSION GET "${holovibes_toolchain}" msvcToolset)
string(JSON VCPKG_CMAKE_SYSTEM_VERSION GET "${holovibes_toolchain}" windowsSdk)
set(VCPKG_HASH_ADDITIONAL_FILES "${CMAKE_CURRENT_LIST_DIR}/../toolchain.json")
