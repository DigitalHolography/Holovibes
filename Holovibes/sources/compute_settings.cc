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

#include "logger.hh"

namespace holovibes::api
{

void after_load_checks()
{
    if (api::detail::get_value<Filter2D>().inner_radius >= api::detail::get_value<Filter2D>().outer_radius)
        api::detail::change_value<Filter2D>()->inner_radius = api::detail::get_value<Filter2D>().inner_radius - 1;
    if (api::detail::get_value<TimeTransformationSize>() < 1)
        api::detail::set_value<TimeTransformationSize>(1);

    // TODO: Check convolution type if it  exists (when it will be added to cd)
    if (api::detail::get_value<ViewAccuP>().start >= api::detail::get_value<TimeTransformationSize>())
        api::detail::change_value<ViewAccuP>()->start = 0;
    if (api::detail::get_value<ViewAccuQ>().start >= api::detail::get_value<TimeTransformationSize>())
        api::detail::change_value<ViewAccuQ>()->start = 0;
    if (api::detail::get_value<ContrastThreshold>().frame_index_offset >
        api::detail::get_value<TimeTransformationSize>() - 1)
        api::detail::change_value<ContrastThreshold>()->frame_index_offset =
            (api::detail::get_value<TimeTransformationSize>() - 1);
}

void load_compute_settings(const std::string& json_path)
{
    LOG_FUNC(json_path);
    if (json_path.empty())
        return;

    std::ifstream ifs(json_path);
    auto j_cs = json::parse(ifs);

    auto compute_settings = ComputeSettings();
    try
    {
        from_json(j_cs, compute_settings);
    }
    catch (const std::exception&)
    {
        LOG_ERROR("{} is an invalid compute settings", json_path);
        return;
    }

    compute_settings.Load();
    compute_settings.Dump("cli_load_compute_settings");

    LOG_INFO("Compute settings loaded from : {}", json_path);

    after_load_checks();
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
