#include "enum_theme.hh"
#include "API.hh"
#include "internals_struct.hh"
#include "compute_settings_struct.hh"
#include <iomanip>

#include "logger.hh"

namespace holovibes::api
{

void after_load_checks()
{
    if (GSH::instance().get_filter2d_n1() >= GSH::instance().get_filter2d_n2())
        GSH::instance().set_filter2d_n1(GSH::instance().get_filter2d_n1() - 1);
    if (GSH::instance().get_time_transformation_size() < 1)
        GSH::instance().set_time_transformation_size(1);
    // TODO: Check convolution type if it  exists (when it will be added to cd)
    if (GSH::instance().get_p().index >= GSH::instance().get_time_transformation_size())
        GSH::instance().set_p_index(0);
    if (GSH::instance().get_q().index >= GSH::instance().get_time_transformation_size())
        GSH::instance().set_q_index(0);
    if (GSH::instance().get_cuts_contrast_p_offset() > GSH::instance().get_time_transformation_size() - 1)
        GSH::instance().set_cuts_contrast_p_offset(GSH::instance().get_time_transformation_size() - 1);
}

void load_compute_settings(const std::string& json_path)
{
    LOG_FUNC(main, json_path);
    if (json_path.empty())
        return;

    std::ifstream ifs(json_path);
    auto j_cs = json::parse(ifs);

    auto compute_settings = ComputeSettings();
    from_json(j_cs, compute_settings);
    compute_settings.Load();

    LOG_INFO(main, "Compute settings loaded from : {}", json_path);

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
    LOG_FUNC(main, json_path);

    if (json_path.empty())
        return;

    std::ofstream file(json_path);
    file << std::setw(1) << compute_settings_to_json();

    LOG_DEBUG(main, "Compute settings overwritten at : {}", json_path);
}
} // namespace holovibes::api
