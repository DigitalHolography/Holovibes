/*! \file compute_settings.hh
 *
 * \brief Regroup all functions related to the manipulation of the compute settings
 */
#pragma once

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

    /*! \brief Returns the compute settings as a json object.
     *
     * \return nlohmann::json The compute settings as a json object
     */
    json compute_settings_to_json() const;

    /*! \brief Load and set settings from a .json file.
     *
     * \param[in] path the absolute path of the .json file.
     */
    void load_compute_settings(const std::string& path) const;

    /*! \brief Change buffer size from a .json file.
     *
     * \param[in] path the absolute path of the .json file.
     */
    void import_buffer(const std::string& path) const;
};

} // namespace holovibes::api