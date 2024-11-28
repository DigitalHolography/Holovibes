/*! \file
 *
 * \brief This file contains the API functions for the Holovibes application. These functions manage input files,
 * camera operations, computation settings, visualization modes, and more. The API functions are used to interact with
 * the Holovibes application from the user interface.
 */

#pragma once

#include <optional>

#include "logger.hh"
#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"
#include "holovibes.hh"
#include "view_panel.hh"
#include "AdvancedSettingsWindow.hh"
#include "holovibes_config.hh"
#include "user_interface_descriptor.hh"
#include "compute_settings_struct.hh"
#include "enum_api_code.hh"

#include <nlohmann/json_fwd.hpp>
using json = ::nlohmann::json;

/*! \brief Return the value of setting T in the holovibes global setting
 * Usage:
 * ```cpp
 * auto value = GET_SETTING(T);
 * ```
 */
#define GET_SETTING(setting) holovibes::Holovibes::instance().get_setting<holovibes::settings::setting>().value

/*! \brief Set the value of setting T in the holovibes global setting to value
 * Usage:
 * ```cpp
 * UPDATE_SETTING(T, value);
 * ```
 */
#define UPDATE_SETTING(setting, value)                                                                                 \
    holovibes::Holovibes::instance().update_setting(holovibes::settings::setting{value})

/*! \brief Update the value.path of setting T in the holovibes global setting to value
 * Usage:
 * ```cpp
 * SET_SETTING(T, path, value);
 *
 * // Is equivalent to
 * auto t = GET_SETTING(T);
 * t.path = value;
 * UPDATE_SETTING(T, t);
 * ```
 */
#define SET_SETTING(type, path, value)                                                                                 \
    {                                                                                                                  \
        auto setting_##type = GET_SETTING(type);                                                                       \
        setting_##type.path = value;                                                                                   \
        UPDATE_SETTING(type, setting_##type);                                                                          \
    }

namespace holovibes::api
{

/*! \brief Get the boundary of frame descriptor
 *
 * \return float boundary
 */
float get_boundary();

/*! \brief Gets the documentation url
 *
 * \return const QUrl& url
 */
const QUrl get_documentation_url();

/*! \brief Gets the credits
 *
 * \return const std::vector<std::string> credits in columns
 */
constexpr std::vector<std::string> get_credits();

/*! \brief Displays information */
void start_information_display();

} // namespace holovibes::api

#include "API.hxx"
#include "composite_api.hh"
#include "record_api.hh"
#include "input_api.hh"
#include "view_api.hh"
#include "filter2d_api.hh"
#include "globalpostprocess_api.hh"
#include "windowpostprocess_api.hh"
#include "contrast_api.hh"
#include "compute_api.hh"
#include "transform_api.hh"

#include "compute_settings.hh"
