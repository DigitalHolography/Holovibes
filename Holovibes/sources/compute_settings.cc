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
    if (GSH::instance().get_value<Filter2DN1>() >= GSH::instance().get_value<Filter2DN2>())
        GSH::instance().set_value<Filter2DN1>(GSH::instance().get_value<Filter2DN1>() - 1);
    if (GSH::instance().get_value<TimeTransformationSize>() < 1)
        GSH::instance().set_value<TimeTransformationSize>(1);
    // TODO: Check convolution type if it  exists (when it will be added to cd)
    if (GSH::instance().get_value<ViewAccuP>().index >= GSH::instance().get_value<TimeTransformationSize>())
        GSH::instance().change_value<ViewAccuP>()->set_index(0);
    if (GSH::instance().get_value<ViewAccuQ>().index >= GSH::instance().get_value<TimeTransformationSize>())
        GSH::instance().change_value<ViewAccuQ>()->set_index(0);
    if (GSH::instance().get_value<CutsContrastPOffset>() > GSH::instance().get_value<TimeTransformationSize>() - 1)
        GSH::instance().set_value<CutsContrastPOffset>(GSH::instance().get_value<TimeTransformationSize>() - 1);
}

void load_compute_settings(const std::string& json_path)
{
    LOG_FUNC(main, json_path);
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
        LOG_ERROR(main, "{} is an invalid compute settings", json_path);
        return;
    }

    compute_settings.Load();
    compute_settings.Dump("cli_load_compute_settings");

    LOG_INFO(main, "Compute settings loaded from : {}", json_path);

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
    LOG_FUNC(main, json_path);

    if (json_path.empty())
        return;

    std::ofstream file(json_path);
    file << std::setw(1) << compute_settings_to_json();

    LOG_DEBUG(main, "Compute settings overwritten at : {}", json_path);
}

} // namespace holovibes::api
