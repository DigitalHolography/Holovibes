/*! \file
 *
 * \brief Contains some constant values needed by the program and in relation with the device.
 */

#pragma once

#include <string>
#include <filesystem>

namespace holovibes::settings
{
// Very ugly to redefine the version in the "camera" folder but we have no choice if we want to define it only once.
#define __HOLOVIBES_VERSION__ "14.7.2"
#define __APPNAME__ "Holovibes"

#define __CAMERAS_CONFIG_FOLDER__ "cameras_config"
#define __PRESET_FOLDER__ "preset"

#define __CONVOLUTION_KERNEL_FOLDER__ "convolution_kernels"
#define __INPUT_FILTER_FOLDER__ "input_filters"
#define __SHADER_FOLDER__ "shaders"

#define __CAMERAS_CONFIG_REFERENCE__ (std::filesystem::path("AppData\\cameras_config"))
#define __PRESET_REFERENCE__ (std::filesystem::path("AppData\\preset"))
#define __CONVOLUTION_KERNEL_REFERENCE__ (std::filesystem::path("AppData\\convolution_kernels"))
#define __INPUT_FILTER_REFERENCE__ (std::filesystem::path("AppData\\input_filters"))
#define __SHADER_REFERENCE__ (std::filesystem::path("AppData\\shaders"))

// AppData location changes depending on Release or Debug mode
#ifdef NDEBUG // Release mode (User AppData)

#define __APPDATA_HOLOVIBES_FOLDER__ (std::filesystem::path(getenv("APPDATA")) / __APPNAME__)
#define __CONFIG_FOLDER__ (__APPDATA_HOLOVIBES_FOLDER__ / __HOLOVIBES_VERSION__)

#else // Debug mode (Local AppData)

#define __APPDATA_HOLOVIBES_FOLDER__ (std::filesystem::path("AppData"))
#define __CONFIG_FOLDER__ (__APPDATA_HOLOVIBES_FOLDER__)

#endif

#define __CONVOLUTION_KERNEL_FOLDER_PATH__ (__CONFIG_FOLDER__ / __CONVOLUTION_KERNEL_FOLDER__)
#define __INPUT_FILTER_FOLDER_PATH__ (__CONFIG_FOLDER__ / __INPUT_FILTER_FOLDER__)
#define __SHADER_FOLDER_PATH__ (__CONFIG_FOLDER__ / __SHADER_FOLDER__)
#define __CAMERAS_CONFIG_FOLDER_PATH__ (__CONFIG_FOLDER__ / __CAMERAS_CONFIG_FOLDER__)
#define __PRESET_FOLDER_PATH__ (__CONFIG_FOLDER__ / __PRESET_FOLDER__)

} // namespace holovibes::settings
