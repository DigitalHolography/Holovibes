/*! \file compute_settings.hh
 *
 * \brief Regroup all functions related to the manipulation of the compute settings
 */
#pragma once

#include "enum_api_code.hh"
#include "holovibes_config.hh"

#include <nlohmann/json_fwd.hpp>

using json = ::nlohmann::json;

namespace holovibes::api
{

class ComputeSettingsApi
{

  public:
    /*! \brief Saves the current state of holovibes in the given file.
     *
     * \param[in] path The absolute path of the .json file where the settings will be saved.
     */
    void save_compute_settings(const std::string& path = ::holovibes::settings::compute_settings_filepath) const;

    /*! \brief Load and set settings from a .json file. The function will also apply conversion patches if needed.
     *
     * \param[in] path the absolute path of the .json file.
     *
     * \return ApiCode the status of the operation: OK if the file was loaded successfully, NO_IN_DATA if the file was
     * not found, FAILURE if the file was invalid.
     */
    ApiCode load_compute_settings(const std::string& path) const;

    /*! \brief Load and set settings from a json object. The function will also apply conversion patches if needed.
     *
     * \param[in] j_cs The json object containing the settings.
     *
     * \return ApiCode the status of the operation: OK if the file was loaded successfully, NO_IN_DATA if the json is
     * empty, FAILURE if the json was invalid.
     */
    ApiCode load_compute_settings(json& j_cs) const;

    /*! \brief Change buffer size from a .json file.
     *
     * \param[in] path the absolute path of the .json file.
     */
    void import_buffer(const std::string& path) const;

    /*! \brief Returns the compute settings as a json object.
     *
     * \return nlohmann::json The compute settings as a json object
     */
    json compute_settings_to_json() const;
};

} // namespace holovibes::api