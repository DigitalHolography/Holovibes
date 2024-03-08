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

void after_load_checks()
{
    auto tts = api::get_time_transformation_size();

    if (api::get_filter2d_n1() >= api::get_filter2d_n2())
        api::set_filter2d_n1(api::get_filter2d_n2() - 1);
    if (tts < 1)
        api::set_time_transformation_size(1);
    // TODO: Check convolution type if it  exists (when it will be added to cd)
    if (holovibes::api::get_p_index() >= tts)
        api::set_p_index(tts - 1);
    if (api::get_q().start >= tts)
        api::set_q_index(tts - 1);
    if (api::get_cuts_contrast_p_offset() > tts - 1)
        api::set_cuts_contrast_p_offset(tts - 1);
}

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


    compute_settings.Load();
    compute_settings.Dump("cli_load_compute_settings");

    LOG_INFO("Compute settings loaded from : {}", json_path);

    after_load_checks();
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
