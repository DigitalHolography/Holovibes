/*! \file
 *
 * \brief Contains functions related to the compute settings
 *
 */

#include "enum_theme.hh"
#include "API.hh"
#include "internals_struct.hh"
#include "compute_settings_struct.hh"
#include "global_state_holder.hh"
#include <iomanip>
#include <spdlog/spdlog.h>

#include "logger.hh"

namespace holovibes::api
{

void load_compute_settings(const std::string& json_path)
{
    LOG_FUNC(json_path);
    if (json_path.empty())
    {
        LOG_WARN("Configuration file not found.");
        return;
    }

    std::ifstream ifs(json_path);
    auto j_cs = json::parse(ifs);

    auto compute_settings = ComputeSettings();
    try
    {
        from_json(j_cs, compute_settings);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("{} is an invalid compute settings", json_path);
        throw std::exception(e);
    }

    compute_settings.Assert();
    compute_settings.Load();
    compute_settings.Dump("cli_load_compute_settings");

    LOG_INFO("Compute settings loaded from : {}", json_path);
    pipe_refresh();
}

void import_buffer(const std::string& json_path)
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
    pipe_refresh();
}

// clang-format off

json compute_settings_to_json()
{
   auto compute_settings = ComputeSettings();
   compute_settings.Update();
   json new_footer;
   to_json(new_footer, compute_settings);
   return new_footer;
}

// clang-format on

void save_compute_settings(const std::string& json_path)
{
    LOG_FUNC(json_path);

    if (json_path.empty())
        return;

    std::ofstream file(json_path);
    file << std::setw(1) << compute_settings_to_json();

    LOG_DEBUG("Compute settings overwritten at : {}", json_path);
}
} // namespace holovibes::api
