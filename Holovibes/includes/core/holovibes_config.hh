/*! \file
 *
 * \brief Contains some constant values needed by the program and in relation with the device.
 */
#pragma once

#include "camera_config.hh"

// Avoid conflict with std::max and std::min (min and max are defined in windows.h)
#include <Windows.h>
#undef max
#undef min

#include <filesystem>

namespace holovibes::settings
{

// Macro wrapper that can be used to craft a relative path from the executable path
#define __GET_EXE_DIR__(type)                                                                                          \
    (                                                                                                                  \
        []                                                                                                             \
        {                                                                                                              \
            type path[MAX_PATH];                                                                                       \
            HMODULE hmodule = GetModuleHandle(NULL);                                                                   \
            if (hmodule != NULL)                                                                                       \
            {                                                                                                          \
                GetModuleFileName(hmodule, path, (sizeof(path)));                                                      \
                std::filesystem::path p(path);                                                                         \
                return p.parent_path();                                                                                \
            }                                                                                                          \
            throw std::runtime_error("Failed to find executable dir");                                                 \
        }())

#ifdef UNICODE

#define GET_EXE_DIR __GET_EXE_DIR__(wchar_t)

#else

#define GET_EXE_DIR __GET_EXE_DIR__(char)

#endif

// Macro wrapper that can be used to craft a relative path from the executable path
#define RELATIVE_PATH(PATH) (GET_EXE_DIR / (PATH))

#define __COMPUTE_CONFIG_FILENAME__ "compute_settings.json"
#define __USER_CONFIG_FILENAME__ "user_settings.json"
#define __LOGS_DIR__ "logs"
#define __PATCH_JSON_DIR__ "patch"
#define __BENCHMARK_FOLDER__ "benchmark"
const static std::string compute_settings_filepath =
    RELATIVE_PATH(__CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string user_settings_filepath = RELATIVE_PATH(__CONFIG_FOLDER__ / __USER_CONFIG_FILENAME__).string();
const static std::string logs_dirpath = RELATIVE_PATH(__CONFIG_FOLDER__ / __LOGS_DIR__).string();
const static std::string patch_dirpath = RELATIVE_PATH(__CONFIG_FOLDER__ / __PATCH_JSON_DIR__).string();
const static std::string benchmark_dirpath = RELATIVE_PATH(__CONFIG_FOLDER__ / __BENCHMARK_FOLDER__).string();
} // namespace holovibes::settings
