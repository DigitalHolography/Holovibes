/*! \file
 *
 * \brief Contains some constant values needed by the program and in relation with the device.
 */
#pragma once

#include "camera_config.hh"
#include <Windows.h>
#include <filesystem>

#include <spdlog/spdlog.h>

namespace holovibes::settings
{

#ifdef UNICODE
#define GET_EXE_DIR()                                                                                                  \
    []                                                                                                                 \
    {                                                                                                                  \
        wchar_t path[MAX_PATH];                                                                                        \
        HMODULE hmodule = GetModuleHandle(NULL);                                                                       \
        if (hmodule != NULL)                                                                                           \
        {                                                                                                              \
            GetModuleFileName(hmodule, path, (sizeof(path)));                                                          \
            std::filesystem::path p(path);                                                                             \
            return p.parent_path().string();                                                                           \
        }                                                                                                              \
        throw std::runtime_error("Failed to find executable dir");                                                     \
    }()

#else
#define GET_EXE_DIR                                                                                                    \
    []                                                                                                                 \
    {                                                                                                                  \
        char path[MAX_PATH];                                                                                           \
        HMODULE hmodule = GetModuleHandle(NULL);                                                                       \
        if (hmodule != NULL)                                                                                           \
        {                                                                                                              \
            GetModuleFileName(hmodule, path, (sizeof(path)));                                                          \
            std::filesystem::path p(path);                                                                             \
            return p.parent_path().string();                                                                           \
        }                                                                                                              \
        throw std::runtime_error("Failed to find executable dir");                                                     \
    }()
#endif

#define __COMPUTE_CONFIG_FILENAME__ "compute_settings.json"
#define __USER_CONFIG_FILENAME__ "user_settings.json"
#define __LOGS_DIR__ "logs"
#define __PATCH_JSON_DIR__ "patch"
#define __BENCHMARK_FOLDER__ "benchmark"
const static std::string compute_settings_filepath =
    (GET_EXE_DIR / __CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string user_settings_filepath = (GET_EXE_DIR / __CONFIG_FOLDER__ / __USER_CONFIG_FILENAME__).string();
const static std::string logs_dirpath = (GET_EXE_DIR / __CONFIG_FOLDER__ / __LOGS_DIR__).string();
const static std::string patch_dirpath = (GET_EXE_DIR / __CONFIG_FOLDER__ / __PATCH_JSON_DIR__).string();
const static std::string benchmark_dirpath = (GET_EXE_DIR / __CONFIG_FOLDER__ / __BENCHMARK_FOLDER__).string();
} // namespace holovibes::settings
