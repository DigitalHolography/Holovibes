#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

void load_image_rendering(const json& data);
void load_view(const json& data);
void load_composite(const json& data);
void load_advanced(const json& data);
void json_to_compute_settings(const json& data);
void after_load_checks();
void load_compute_settings(const std::string& json_path);
json compute_settings_to_json();
void save_compute_settings(const std::string& json_path = holovibes::settings::compute_settings_filepath);

} // namespace holovibes::api
