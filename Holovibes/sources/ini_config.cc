#include "ini_config.hh"
#include "global_state_holder.hh"

namespace holovibes::ini
{
void load_image_rendering(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    cd.compute_mode = static_cast<Computation>(
        ptree.get<int>("image_rendering.image_mode", static_cast<int>(cd.compute_mode.load())));
    cd.filter2d_enabled = ptree.get<bool>("image_rendering.filter2d_enabled", cd.filter2d_enabled);
    cd.space_transformation = static_cast<SpaceTransformation>(
        ptree.get<int>("image_rendering.space_transformation", static_cast<int>(cd.space_transformation.load())));
    cd.time_transformation = static_cast<TimeTransformation>(
        ptree.get<int>("image_rendering.time_transformation", static_cast<int>(cd.time_transformation.load())));

    cd.lambda = ptree.get<float>("image_rendering.lambda", cd.lambda);
    cd.zdistance = ptree.get<float>("image_rendering.z_distance", cd.zdistance);
    cd.convolution_enabled = ptree.get<bool>("image_rendering.convolution_enabled", cd.convolution_enabled);
    // TODO: Think about how to store the type. Some new convolutions type might be added in AppData
    // cd.convolution_type = ptree.get("image_rendering.convolution_type", cd.convolution_enabled);
    cd.divide_convolution_enabled =
        ptree.get<bool>("image_rendering.divide_convolution_enabled", cd.divide_convolution_enabled);
}

void load_view(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    // FIXME: Add a call to set_view_mode(), this fuunction is currently in mainwindow
    cd.img_type = static_cast<ImgType>(ptree.get<int>("view.view_type", static_cast<int>(cd.img_type.load())));
    // Add unwrap_2d
    cd.time_transformation_cuts_enabled =
        ptree.get<bool>("view.time_transformation_cuts", cd.time_transformation_cuts_enabled);
    cd.fft_shift_enabled = ptree.get<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);
    cd.lens_view_enabled = ptree.get<bool>("view.lens_view_enabled", cd.lens_view_enabled);
    cd.raw_view_enabled = ptree.get<bool>("view.raw_view_enabled", cd.raw_view_enabled);

    auto xypq_load = [&](const std::string name, View_Accu& view) {
        view.accu_level = ptree.get<short>("view." + name + "_accu_level", view.accu_level);
    };
    auto xy_load = [&](const std::string name, View_XY& view) {
        view.cuts = ptree.get<ushort>("view." + name + "_cuts", view.cuts);
        xypq_load(name, view);
    };
    auto pq_load = [&](const std::string name, View_PQ& view) {
        view.index = ptree.get<ushort>("view." + name + "_index", view.index);
        xypq_load(name, view);
    };

    xy_load("x", cd.x);
    xy_load("y", cd.y);
    pq_load("p", cd.p);
    pq_load("q", cd.q);

    cd.renorm_enabled = ptree.get<bool>("view.renorm_enabled", cd.renorm_enabled);

    cd.reticle_view_enabled = ptree.get<bool>("view.reticle_view_enabled", cd.reticle_view_enabled);
    cd.reticle_scale = ptree.get<float>("view.reticle_scale", cd.reticle_scale);

    auto xyzf_load = [&](const std::string name, View_Window& view) {
        view.log_scale_slice_enabled =
            ptree.get<bool>("view." + name + "_log_scale_enabled", view.log_scale_slice_enabled);

        view.contrast_enabled = ptree.get<bool>("view." + name + "_contrast_enabled", view.contrast_enabled);
        view.contrast_auto_refresh =
            ptree.get<bool>("view." + name + "_auto_contrast_enabled", view.contrast_auto_refresh);
        view.contrast_invert = ptree.get<bool>("view." + name + "_invert_enabled", view.contrast_invert);
        view.contrast_min = ptree.get<float>("view." + name + "_contrast_min", view.contrast_min);
        view.contrast_max = ptree.get<float>("view." + name + "_contrast_max", view.contrast_max);
    };

    auto xyz_load = [&](const std::string name, View_XYZ& view) {
        view.flip_enabled = ptree.get<bool>("view." + name + "_flip_enabled", view.flip_enabled);
        view.rot = ptree.get<float>("view." + name + "_rot", view.rot);

        view.img_accu_level = ptree.get<ushort>("view." + name + "_img_accu_level", view.img_accu_level);

        xyzf_load(name, view);
    };

    xyz_load("xy", cd.xy);
    xyz_load("xz", cd.xz);
    xyz_load("yz", cd.yz);
    xyzf_load("filter2d", cd.filter2d);
}

void load_composite(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    // cd.composite_kind =
    //     static_cast<CompositeKind>(ptree.get<int>("composite.mode", static_cast<int>(cd.composite_kind.load())));
    cd.composite_auto_weights = ptree.get<bool>("composite.auto_weights_enabled", cd.composite_auto_weights);

    auto p_load = [&](const std::string& name, Composite_P& p) {
        p.p_min = ptree.get<ushort>("composite." + name + "_p_min", cd.rgb.p_min);
        p.p_max = ptree.get<ushort>("composite." + name + "_p_max", cd.rgb.p_max);
    };

    p_load("rgb", cd.rgb);
    cd.rgb.weight_r = ptree.get<float>("composite.rgb_weight_r", cd.rgb.weight_r);
    cd.rgb.weight_g = ptree.get<float>("composite.rgb_weight_g", cd.rgb.weight_g);
    cd.rgb.weight_b = ptree.get<float>("composite.rgb_weight_b", cd.rgb.weight_b);

    auto hsv_load = [&](const std::string& name, Composite_hsv& s) {
        p_load(name, s);
        s.slider_threshold_min = ptree.get<float>("composite." + name + "_min_value", s.slider_threshold_min);
        s.slider_threshold_max = ptree.get<float>("composite." + name + "_max_value", s.slider_threshold_max);
        s.low_threshold = ptree.get<float>("composite." + name + "_low_threshold", s.low_threshold);
        s.high_threshold = ptree.get<float>("composite." + name + "_high_threshold", s.high_threshold);
    };

    hsv_load("hsv_h", cd.hsv.h);
    cd.hsv.h.blur_enabled = ptree.get<bool>("hsv_h.blur_enabled", cd.hsv.h.blur_enabled);
    cd.hsv.h.blur_kernel_size = ptree.get<uint>("hsv_h.blur_size", cd.hsv.h.blur_kernel_size);

    auto sv_load = [&](const std::string& name, Composite_SV& s) {
        s.p_activated = ptree.get<bool>("composite." + name + "_enabled", s.p_activated);
        hsv_load(name, s);
    };

    sv_load("hsv_s", cd.hsv.s);
    sv_load("hsv_v", cd.hsv.v);
}

void load_advanced(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    cd.file_buffer_size = ptree.get<ushort>("advanced.file_buffer_size", cd.file_buffer_size);
    cd.input_buffer_size = ptree.get<ushort>("advanced.input_buffer_size", cd.input_buffer_size);
    cd.record_buffer_size = ptree.get<ushort>("advanced.record_buffer_size", cd.record_buffer_size);
    cd.output_buffer_size = ptree.get<ushort>("advanced.output_buffer_size", cd.output_buffer_size);
    cd.time_transformation_cuts_output_buffer_size =
        ptree.get<ushort>("advanced.time_transformation_cuts_output_buffer_size",
                          cd.time_transformation_cuts_output_buffer_size);
    cd.display_rate = ptree.get<float>("advanced.display_rate", cd.display_rate);
    cd.filter2d_smooth_low = ptree.get<int>("advanced.filter2d_smooth_low", cd.filter2d_smooth_low);
    cd.filter2d_smooth_high = ptree.get<int>("advanced.filter2d_smooth_high", cd.filter2d_smooth_high);
    cd.contrast_lower_threshold = ptree.get<float>("advanced.contrast_lower_threshold", cd.contrast_lower_threshold);
    cd.contrast_upper_threshold = ptree.get<float>("advanced.contrast_upper_threshold", cd.contrast_upper_threshold);
    cd.renorm_constant = ptree.get<uint>("advanced.renorm_constant", cd.renorm_constant);
    cd.cuts_contrast_p_offset = ptree.get<ushort>("view.cuts_contrast_p_offset", cd.cuts_contrast_p_offset);
}

void after_load_checks(ComputeDescriptor& cd)
{
    if (cd.filter2d_n1 >= cd.filter2d_n2)
        cd.filter2d_n1 = cd.filter2d_n2 - 1;
    // TODO: Check convolution type if it  exists (when it will be added to cd)

    uint time_transformation_size = GSH::instance().time_transformation_size_query().value;

    if (cd.p.index >= time_transformation_size)
        cd.p.index = 0;
    if (cd.q.index >= time_transformation_size)
        cd.q.index = 0;
    if (cd.cuts_contrast_p_offset > time_transformation_size - 1)
        cd.cuts_contrast_p_offset = time_transformation_size - 1;
}

void load_compute_settings(ComputeDescriptor& cd, const std::string& ini_path)
{
    LOG_INFO << "Compute settings loaded from : " << ini_path;

    boost::property_tree::ptree ptree;
    boost::property_tree::ini_parser::read_ini(ini_path, ptree);

    load_image_rendering(ptree, cd);
    load_view(ptree, cd);
    load_composite(ptree, cd);
    load_advanced(ptree, cd);

    GSH::instance().load_ptree(ptree);

    after_load_checks(cd);
}

void save_image_rendering(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd)
{
    ptree.put<int>("image_rendering.image_mode", static_cast<int>(cd.compute_mode.load()));
    ptree.put<bool>("image_rendering.filter2d_enabled", static_cast<int>(cd.filter2d_enabled.load()));
    ptree.put<int>("image_rendering.filter2d_n1", cd.filter2d_n1.load());
    ptree.put<int>("image_rendering.filter2d_n2", cd.filter2d_n2.load());
    ptree.put<int>("image_rendering.space_transformation", static_cast<int>(cd.space_transformation.load()));
    ptree.put<int>("image_rendering.time_transformation", static_cast<int>(cd.time_transformation.load()));
    ptree.put<float>("image_rendering.lambda", cd.lambda);
    ptree.put<float>("image_rendering.z_distance", cd.zdistance);
    ptree.put<bool>("image_rendering.convolution_enabled", cd.convolution_enabled);
    // ptree.put<string>("image_rendering.convolution_type", cd.convolution_type);
    ptree.put<bool>("image_rendering.divide_convolution_enabled", cd.divide_convolution_enabled);
}

void save_view(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd)
{
    ptree.put<int>("view.view_type", static_cast<int>(cd.img_type.load()));
    // ptree.put<bool>("view.unwrap_2d_enabled", cd.unwrap_2d);
    ptree.put<bool>("view.3d_cuts_enabled", cd.time_transformation_cuts_enabled);
    ptree.put<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);
    ptree.put<bool>("view.lens_view_enabled", cd.lens_view_enabled);
    ptree.put<bool>("view.raw_view_enabled", cd.raw_view_enabled);

    auto xypq_save = [&](const std::string& name, const View_Accu& view) {
        ptree.put<short>("view." + name + "_accu_level", view.accu_level);
    };

    auto xy_save = [&](const std::string& name, const View_XY& view) {
        ptree.put<ushort>("view." + name + "_cuts", view.cuts);
        xypq_save(name, view);
    };

    xy_save("x", cd.x);
    xy_save("y", cd.y);

    auto pq_save = [&](const std::string& name, const View_PQ& view) {
        ptree.put<ushort>("view." + name + "_index", view.index);
        xypq_save(name, view);
    };

    pq_save("p", cd.p);
    pq_save("q", cd.q);

    ptree.put<bool>("view.renorm_enabled", cd.renorm_enabled);
    ptree.put<bool>("view.reticle_view_enabled", cd.reticle_view_enabled);
    ptree.put<float>("view.reticle_scale", cd.reticle_scale);

    auto xyzf_save = [&](const std::string& name, const View_Window& view) {
        ptree.put<bool>("view." + name + "_log_scale_enabled", view.log_scale_slice_enabled);
        ptree.put<bool>("view." + name + "_contrast_enabled", view.contrast_enabled);
        ptree.put<bool>("view." + name + "_contrast_auto_enabled", view.contrast_auto_refresh);
        ptree.put<bool>("view." + name + "_contrast_invert_enabled", view.contrast_invert);
        ptree.put<float>("view." + name + "_contrast_min", view.contrast_min);
        ptree.put<float>("view." + name + "_contrast_max", view.contrast_max);
    };

    auto xyz_save = [&](const std::string& name, const View_XYZ& view) {
        ptree.put<bool>("view." + name + "_flip_enabled", view.flip_enabled);
        ptree.put<int>("view." + name + "_rot", view.rot);
        ptree.put<ushort>("view." + name + "_img_accu_level", view.img_accu_level);

        xyzf_save(name, view);
    };

    xyz_save("xy", cd.xy);
    xyz_save("xz", cd.xz);
    xyz_save("yz", cd.yz);
    xyzf_save("filter2d", cd.filter2d);
}

void save_composite(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd)
{
    ptree.put<int>("composite.mode", static_cast<int>(cd.composite_kind.load()));
    ptree.put<bool>("composite.auto_weights_enabled", cd.composite_auto_weights);

    auto p_save = [&](const std::string& name, const Composite_P& p) {
        ptree.put<ushort>("composite." + name + "_p_min", cd.rgb.p_min);
        ptree.put<ushort>("composite." + name + "_p_max", cd.rgb.p_max);
    };

    p_save("rgb", cd.rgb);
    ptree.put<float>("composite.rgb_weight_r", cd.rgb.weight_r);
    ptree.put<float>("composite.rgb_weight_g", cd.rgb.weight_g);
    ptree.put<float>("composite.rgb_weight_b", cd.rgb.weight_b);

    auto hsv_save = [&](const std::string& name, const Composite_hsv& s) {
        p_save(name, s);
        ptree.put<float>("composite." + name + "_min_value", s.slider_threshold_min);
        ptree.put<float>("composite." + name + "_max_value", s.slider_threshold_max);
        ptree.put<float>("composite." + name + "_low_threshold", s.low_threshold);
        ptree.put<float>("composite." + name + "_high_threshold", s.high_threshold);
    };

    hsv_save("hsv_h", cd.hsv.h);
    ptree.put<bool>("composite.hsv_h_blur_enabled", cd.hsv.h.blur_enabled);
    ptree.put<ushort>("composite.hsv_h_blur_size", cd.hsv.h.blur_kernel_size);

    auto sv_save = [&](const std::string& name, const Composite_SV& s) {
        ptree.put<bool>("composite." + name + "_enabled", s.p_activated);
        hsv_save(name, s);
    };

    sv_save("hsv_s", cd.hsv.s);
    sv_save("hsv_v", cd.hsv.v);
}

void save_advanced(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd)
{
    ptree.put<uint>("advanced.file_buffer_size", cd.file_buffer_size);
    ptree.put<uint>("advanced.input_buffer_size", cd.input_buffer_size);
    ptree.put<uint>("advanced.record_buffer_size", cd.record_buffer_size);
    ptree.put<uint>("advanced.output_buffer_size", cd.output_buffer_size);
    ptree.put<uint>("advanced.time_transformation_cuts_output_buffer_size",
                    cd.time_transformation_cuts_output_buffer_size);
    ptree.put<ushort>("advanced.display_rate", static_cast<ushort>(cd.display_rate));
    ptree.put<int>("advanced.filter2d_smooth_low", cd.filter2d_smooth_low.load());
    ptree.put<int>("advanced.filter2d_smooth_high", cd.filter2d_smooth_high.load());
    ptree.put<float>("advanced.contrast_lower_threshold", cd.contrast_lower_threshold);
    ptree.put<float>("advanced.contrast_upper_threshold", cd.contrast_upper_threshold);
    ptree.put<uint>("advanced.renorm_constant", cd.renorm_constant);
    ptree.put<ushort>("advanced.cuts_contrast_p_offset", cd.cuts_contrast_p_offset);
}

void save_compute_settings(const ComputeDescriptor& cd, const std::string& ini_path)
{
    boost::property_tree::ptree ptree;

    save_image_rendering(ptree, cd);
    save_view(ptree, cd);
    save_composite(ptree, cd);
    save_advanced(ptree, cd);

    GSH::instance().dump_ptree(ptree);

    boost::property_tree::write_ini(ini_path, ptree);

    LOG_INFO << "Compute settings overwritten at : " << ini_path;
}
} // namespace holovibes::ini
