/*! \file
 *
 * \brief Contains some constant values needed by the program and in relation with the device.
 */
#pragma once

#include "camera_config.hh"
#include <Windows.h>
#include <filesystem>

// FIXME: get rid of this duplicate get_exe_dir, find a way to import the original one in tools.hh or move it to a
// a place from which we can import it easily
static std::string get_exe_dir()
{
#ifdef UNICODE
    wchar_t path[MAX_PATH];
#else
    char path[MAX_PATH];
#endif
    HMODULE hmodule = GetModuleHandle(NULL);
    if (hmodule != NULL)
    {
        GetModuleFileName(hmodule, path, (sizeof(path)));
        std::filesystem::path p(path);
        return p.parent_path().string();
    }

    // FIXME: add logging
    // Logger::camera()->error("Failed to find executable dir");
    throw std::runtime_error("Failed to find executable dir");
}

namespace holovibes::settings
{
#define __COMPUTE_CONFIG_FILENAME__ "compute_settings.json"
#define __USER_CONFIG_FILENAME__ "user_settings.json"
#define __LOGS_DIR__ "logs"
#define __PATCH_JSON_DIR__ "patch"
#define __BENCHMARK_FOLDER__ "benchmark"
const static std::string compute_settings_filepath =
    (get_exe_dir() / __CONFIG_FOLDER__ / __COMPUTE_CONFIG_FILENAME__).string();
const static std::string user_settings_filepath =
    (get_exe_dir() / __CONFIG_FOLDER__ / __USER_CONFIG_FILENAME__).string();
const static std::string logs_dirpath = (get_exe_dir() / __CONFIG_FOLDER__ / __LOGS_DIR__).string();
const static std::string patch_dirpath = (get_exe_dir() / __CONFIG_FOLDER__ / __PATCH_JSON_DIR__).string();
const static std::string benchmark_dirpath = (get_exe_dir() / __CONFIG_FOLDER__ / __BENCHMARK_FOLDER__).string();
} // namespace holovibes::settings
