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
#define __HOLOVIBES_VERSION__ "13.4.0"
#define __APPNAME__ "Holovibes"

#define __CAMERAS_CONFIG_FOLDER__ "cameras_config"
#define __PRESET_FOLDER__ "preset"
#define __CONVOLUTION_KERNEL_FOLDER__ "convolution_kernels"
#define __INPUT_FILTER_FOLDER__ "input_filters"
#define __SHADER_FOLDER__ "shaders"

#define __GET_PATH__(PATH) (std::filesystem::absolute(std::filesystem::path(PATH)))
/*
#define __CAMERAS_CONFIG_REFERENCE__ (__GET_PATH__("build/bin/AppData/cameras_config"))
#define __PRESET_REFERENCE__ (__GET_PATH__("build/bin/AppData/preset"))
#define __CONVOLUTION_KERNEL_REFERENCE__ (__GET_PATH__("build/bin/convolution_kernels"))
#define __INPUT_FILTER_REFERENCE__ (__GET_PATH__("build/bin/input_filters"))
*/
#ifdef NDEBUG

#define __APPDATA_HOLOVIBES_FOLDER__ (std::filesystem::path(getenv("APPDATA")) / __APPNAME__)
#define __CONFIG_FOLDER__ (__APPDATA_HOLOVIBES_FOLDER__ / __HOLOVIBES_VERSION__)

#define __CAMERAS_CONFIG_FOLDER_PATH__ (__CONFIG_FOLDER__ / __CAMERAS_CONFIG_FOLDER__)
#define __PRESET_FOLDER_PATH__ (__CONFIG_FOLDER__ / __PRESET_FOLDER__)
#define __CONVOLUTION_KERNEL_FOLDER_PATH__ (__CONFIG_FOLDER__ / __CONVOLUTION_KERNEL_FOLDER__)
#define __INPUT_FILTER_FOLDER_PATH__ (__CONFIG_FOLDER__ / __INPUT_FILTER_FOLDER__)
#define __SHADER_FOLDER_PATH__ (__CONFIG_FOLDER__ / __SHADER_FOLDER__)

#else

#define __APPDATA_HOLOVIBES_FOLDER__ (std::filesystem::absolute(std::filesystem::path("build/bin/AppData/")))
#define __CONFIG_FOLDER__ (__APPDATA_HOLOVIBES_FOLDER__)

#define __CAMERAS_CONFIG_FOLDER_PATH__ (__GET_PATH__("Camera/configs")) //(__CAMERAS_CONFIG_REFERENCE__)
#define __PRESET_FOLDER_PATH__ (__GET_PATH__("preset"))                 //(__PRESET_REFERENCE__)
#define __CONVOLUTION_KERNEL_FOLDER_PATH__                                                                             \
    (__GET_PATH__("Holovibes/convolution_kernels"))                            //(__CONVOLUTION_KERNEL_REFERENCE__)
#define __INPUT_FILTER_FOLDER_PATH__ (__GET_PATH__("Holovibes/input_filters")) //(__INPUT_FILTER_REFERENCE__)
#define __SHADER_FOLDER_PATH__ (__GET_PATH__("Holovibes/shaders"))

#endif

} // namespace holovibes::settings
