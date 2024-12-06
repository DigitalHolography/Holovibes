/*! \file common_api.hh
 *
 * \brief  Common API functions
 */

#pragma once

#include <optional>

#include "logger.hh"
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
 *
 * \param[in] setting The setting to get
 * \return The value of the setting T
 */
#define GET_SETTING(setting) holovibes::Holovibes::instance().get_setting<holovibes::settings::setting>().value

/*! \brief Set the value of setting T in the holovibes global setting to value
 * Usage:
 * ```cpp
 * UPDATE_SETTING(T, value);
 * ```
 *
 * \param[in] setting The setting to update
 * \param[in] value The new value of the setting
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
 *
 * \param[in] setting The setting to update
 * \param[in] path The path of the setting to update (e.g. value, min, max, etc.)
 * \param[in] value The new value of the setting
 */
#define SET_SETTING(type, path, value)                                                                                 \
    {                                                                                                                  \
        auto setting_##type = GET_SETTING(type);                                                                       \
        setting_##type.path = value;                                                                                   \
        UPDATE_SETTING(type, setting_##type);                                                                          \
    }

namespace holovibes::api
{
class Api;

class IApi
{
  public:
    // Take reference to parent in ctor
    IApi() {}

    void set_api(Api* api) { api_ = api; }

    virtual ~IApi() = default;

  protected:
    Api* api_;
};

} // namespace holovibes::api