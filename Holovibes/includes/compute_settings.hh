/*! \file
 *
 * \brief Regroup all functions related to the manipulation of the compute settings
 */
#pragma once

#include "common_api.hh"

namespace holovibes::api
{
/*! \brief Saves the current state of holovibes
 *
 * \param path The location of the .json file saved
 */
void save_compute_settings(const std::string& path = ::holovibes::settings::compute_settings_filepath);

json compute_settings_to_json();

/*! \brief Setups program from .json file
 *
 * \param path the path where the .json file is
 */
void load_compute_settings(const std::string& path);

/*! \brief Change buffer siwe from .json file
 *
 * \param path the path where the .json file is
 */
void import_buffer(const std::string& path);

} // namespace holovibes::api