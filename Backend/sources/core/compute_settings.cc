/*! \file
 *
 * \brief Contains functions related to the compute settings
 *
 */
#include "compute_settings.hh"

#include <iomanip>

#include "API.hh"
#include "compute_settings_converter.hh"
#include "compute_settings_struct.hh"
#include "enum_theme.hh"
#include "internals_struct.hh"
#include "logger.hh"

namespace holovibes::api
{

/*! \brief Merge two json objects
 *
 * \param[in] base_json The json object to merge into
 * \param[in] update_json The json object to merge from
 */
void merge_json(json& base_json, const json& update_json)
{
    for (auto& [key, value] : update_json.items())
    {
        if (!base_json.contains(key))
            throw std::runtime_error("Error: Key '" + key + "' found in update_json but not in base_json");

        if (base_json[key].is_object() && value.is_object())
            merge_json(base_json[key], value);
        else
            base_json[key] = value;
    }
}

ApiCode ComputeSettingsApi::load_compute_settings(const std::string& json_path) const
{
    LOG_FUNC(json_path);

    if (json_path.empty())
    {
        LOG_WARN("Configuration file not found.");
        return ApiCode::NO_IN_DATA;
    }

    nlohmann::json j_cs;

    // Parse the file
    try
    {
        std::ifstream ifs(json_path);
        ifs >> j_cs;
    }
    catch (const std::exception&)
    {
        LOG_ERROR("File not found or invalid json file : {}", json_path);
        return ApiCode::FAILURE;
    }

    return load_compute_settings(j_cs);
}

ApiCode ComputeSettingsApi::load_compute_settings(json& j_cs) const
{
    LOG_FUNC(j_cs.dump());

    if (j_cs.empty())
    {
        LOG_WARN("Empty compute settings.");
        return ApiCode::NO_IN_DATA;
    }

    ApiCode res = version::ComputeSettingsConverter::convert_compute_settings(j_cs);
    if (res != ApiCode::OK)
        return res;

    auto compute_settings = ComputeSettings();
    compute_settings.Update();
    json old_one = compute_settings;

    try
    {
        // this allows a compute_settings to not have all the fields and to still work
        merge_json(old_one, j_cs);
        compute_settings = old_one;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Invalid compute settings: {}", e.what());
        return ApiCode::FAILURE;
    }

    compute_settings.Assert();
    compute_settings.Load();

    return ApiCode::OK;
}

void ComputeSettingsApi::import_buffer(const std::string& json_path) const
{

    LOG_FUNC(json_path);
    if (json_path.empty())
    {
        LOG_WARN("Configuration file not found.");
        return;
    }

    std::ifstream ifs(json_path);
    auto j_cs = json::parse(ifs);

    auto advanced_settings = AdvancedSettings();

    auto buffer_settings = AdvancedSettings::BufferSizes();

    try
    {
        from_json(j_cs, buffer_settings);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("{} is an invalid buffer settings", json_path);
        throw std::exception(e);
    }

    buffer_settings.Assert();
    buffer_settings.Load();

    LOG_INFO("Buffer settings loaded from : {}", json_path);
}

// clang-format off

json ComputeSettingsApi::compute_settings_to_json() const
{
   auto compute_settings = ComputeSettings();
   compute_settings.Update();
   json new_footer;
   to_json(new_footer, compute_settings);
   return new_footer;
}

// clang-format on

void ComputeSettingsApi::save_compute_settings(const std::string& json_path) const
{
    LOG_FUNC(json_path);

    if (json_path.empty())
        return;

    std::ofstream file(json_path);
    file << std::setw(1) << compute_settings_to_json();

    LOG_DEBUG("Compute settings overwritten at : {}", json_path);
}
} // namespace holovibes::api
