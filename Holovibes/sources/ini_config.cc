#include "global_state_holder.hh"
#include "API.hh"

namespace holovibes::api
{
void load_image_rendering(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    ////// TODO: Remove checkbox ??
    ////// TODO: Think about how to store the type. Some new convolutions type might be added in AppData
    ////// set_convolution_enabled(ptree.get<bool>("image_rendering.convolution_enabled", cd.convolution_enabled));
    ////// cd.convolution_type( ptree.get("image_rendering.convolution_type", cd.convolution_enabled));
}
void load_view(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd) {}

void after_load_checks(ComputeDescriptor& cd)
{
    if (GSH::instance().get_filter2d_n1() >= GSH::instance().get_filter2d_n2())
        GSH::instance().set_filter2d_n1(GSH::instance().get_filter2d_n2() - 1);
    // TODO: Check convolution type if it  exists (when it will be added to cd)

    uint time_transformation_size = GSH::instance().get_time_transformation_size();

    if (GSH::instance().get_p_index() >= time_transformation_size)
        GSH::instance().set_p_index(0);
    if (GSH::instance().get_q_index() >= time_transformation_size)
        GSH::instance().set_q_index(0);
    if (GSH::instance().get_cuts_contrast_p_offset() > time_transformation_size - 1)
        GSH::instance().set_cuts_contrast_p_offset(time_transformation_size - 1);
}

void load_compute_settings(const std::string& ini_path)
{
    if (ini_path.empty())
        return;

    LOG_INFO << "Compute settings loaded from : " << ini_path;

    boost::property_tree::ptree ptree;
    boost::property_tree::ini_parser::read_ini(ini_path, ptree);

    load_image_rendering(ptree, get_cd());
    load_view(ptree, get_cd());

    GSH::instance().load_ptree(ptree);

    after_load_checks(get_cd());

    pipe_refresh();
}

void save_image_rendering(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd) {}

void save_view(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd)
{
    // ptree.put<bool>("view.unwrap_2d_enabled", cd.unwrap_2d);

    auto pq_save = [&](const std::string& name, const View_PQ& view) {
        ptree.put<ushort>("view." + name + "_index", view.index);
        ptree.put<short>("view." + name + "_accu_level", view.accu_level);
    };
}

void save_advanced(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd) {}

void save_compute_settings(const std::string& ini_path)
{
    if (ini_path.empty())
        return;

    boost::property_tree::ptree ptree;

    save_image_rendering(ptree, get_cd());
    save_view(ptree, get_cd());
    save_advanced(ptree, get_cd());

    GSH::instance().dump_ptree(ptree);

    boost::property_tree::write_ini(ini_path, ptree);

    LOG_INFO << "Compute settings overwritten at : " << ini_path;
}
} // namespace holovibes::api
