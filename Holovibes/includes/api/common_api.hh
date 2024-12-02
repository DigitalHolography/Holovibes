/*! \file
 *
 * \brief This file contains common header and maccros for all APIs files.
 */

#pragma once

#include <optional>

#include "logger.hh"
#include "input_frame_file.hh"
#include "input_frame_file_factory.hh"
#include "holovibes.hh"
#include "holovibes_config.hh"
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