/*! \file API.hh
 *
 * \brief This file contains the API functions for the Holovibes application. These functions manage input files,
 * camera operations, computation settings, visualization modes, and more. The API functions are used to interact with
 * the Holovibes application from the user interface.
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

    void set_api(Api& api) { api_ = api; }

    virtual ~IApi() = default;

  protected:
    Api& api_;
};

} // namespace holovibes::api

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
#include "information_api.hh"

#include "compute_settings.hh"

namespace holovibes::api
{

#define API holovibes::api::Api::instance()

class Api
{

  private:
    // Private ctor
    Api() { init(); }

    void init()
    {
        composite.set_api(*this);
        compute.set_api(*this);
        contrast.set_api(*this);
        filter2d.set_api(*this);
        global_pp.set_api(*this);
        information.set_api(*this);
        input.set_api(*this);
        record.set_api(*this);
        transform.set_api(*this);
        view.set_api(*this);
        window_pp.set_api(*this);
    }

  public:
    // Singleton
    static Api& instance()
    {
        static Api instance;
        return instance;
    }

  public:
    CompositeApi composite;
    ComputeApi compute;
    ContrastApi contrast;
    Filter2dApi filter2d;
    GlobalPostProcessApi global_pp;
    InformationApi information;
    InputApi input;
    RecordApi record;
    TransformApi transform;
    ViewApi view;
    WindowPostProcessApi window_pp;
    ComputeSettingsApi settings;
};

} // namespace holovibes::api