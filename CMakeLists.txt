cmake_minimum_required(VERSION 3.15)

project(Holovibes)

find_package(Boost REQUIRED
    COMPONENTS
    program_options
)

if(CMAKE_TESTING_ENABLED)

# Fetch googletest git repository during CMake configuration
include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        release-1.11.0 # Specify the commit you depend on and update it regularly.
)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

endif()

add_subdirectory(Camera)
add_subdirectory(GPIB)
add_subdirectory(Holovibes)
