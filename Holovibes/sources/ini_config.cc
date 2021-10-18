#include "ini_config.hh"

namespace holovibes::ini
{
void load_ini(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    // Image rendering
    cd.compute_mode = static_cast<Computation>(
        ptree.get<int>("image_rendering.image_mode", static_cast<int>(cd.compute_mode.load())));
    cd.batch_size = ptree.get<ushort>("image_rendering.batch_size", cd.batch_size);
    cd.time_transformation_stride =
        ptree.get<ushort>("image_rendering.time_transformation_stride", cd.time_transformation_stride);
    cd.filter2d_enabled = ptree.get<bool>("image_rendering.filter2d_enabled", cd.filter2d_enabled);
    cd.filter2d_n1 = ptree.get<int>("image_rendering.filter2d_n1", cd.filter2d_n1);
    cd.filter2d_n2 = ptree.get<int>("image_rendering.filter2d_n2", cd.filter2d_n2);
    cd.space_transformation = static_cast<SpaceTransformation>(
        ptree.get<int>("image_rendering.space_transformation", static_cast<int>(cd.space_transformation.load())));
    cd.time_transformation = static_cast<TimeTransformation>(
        ptree.get<int>("image_rendering.time_transformation", static_cast<int>(cd.time_transformation.load())));
    cd.time_transformation_size =
        ptree.get<ushort>("image_rendering.time_transformation_size", cd.time_transformation_size);
    cd.lambda = ptree.get<float>("image_rendering.lambda", cd.lambda);
    cd.zdistance = ptree.get<float>("image_rendering.z_distance", cd.zdistance);
    cd.convolution_enabled = ptree.get<bool>("image_rendering.convolution_enabled", cd.convolution_enabled);
    // TODO: Think about how to store the type. Some new convolutions type might be added in AppData
    // cd.convolution_type = ptree.get("image_rendering.convolution_type", cd.convolution_enabled);
    cd.divide_convolution_enabled =
        ptree.get<bool>("image_rendering.divide_convolution_enabled", cd.divide_convolution_enabled);

    // View
    // FIXME: Add a call to set_view_mode(), this fuunction is currently in mainwindow
    cd.img_type = static_cast<ImgType>(ptree.get<int>("view.view_type", static_cast<int>(cd.img_type.load())));
    // Add unwrap_2d
    cd.time_transformation_cuts_enabled =
        ptree.get<bool>("view.time_transformation_cuts", cd.time_transformation_cuts_enabled);
    cd.fft_shift_enabled = ptree.get<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);
    cd.lens_view_enabled = ptree.get<bool>("view.lens_view_enabled", cd.lens_view_enabled);
    cd.raw_view_enabled = ptree.get<bool>("view.raw_view_enabled", cd.raw_view_enabled);

    auto xypq_get = [&](const std::string name, AccView& view) {
        view.accu_enabled = ptree.get<bool>("view." + name + "_accu_enabled", view.accu_enabled);
        view.accu_level = ptree.get<short>("view." + name + "_accu_level", view.accu_level);
    };
    cd.x.cuts = ptree.get<ushort>("view.x_cuts", cd.x.cuts);
    xypq_get("x", cd.x);
    cd.y.cuts = ptree.get<ushort>("view.y_cuts", cd.y.cuts);
    xypq_get("y", cd.y);
    cd.p.index = ptree.get<ushort>("view.p_index", cd.p.index);
    xypq_get("p", cd.p);
    cd.q.index = ptree.get<ushort>("view.q_index", cd.q.index);
    xypq_get("q", cd.q);

    cd.renorm_enabled = ptree.get<bool>("view.renorm_enabled", cd.renorm_enabled);
    cd.reticle_view_enabled = ptree.get<bool>("view.reticle_view_enabled", cd.reticle_view_enabled);
    cd.reticle_scale = ptree.get<float>("view.reticle_scale", cd.reticle_scale);

    auto xyz_get = [&](const std::string name, XY_XZ_YZ_WindowView& view) {
        view.flip_enabled = ptree.get<bool>(name + ".flip_enabled", view.flip_enabled);
        view.rot = ptree.get<float>(name + ".rot", view.rot);
        view.log_scale_slice_enabled = ptree.get<bool>(name + ".log_scale_enabled", view.log_scale_slice_enabled);
        view.img_acc_slice_enabled = ptree.get<bool>(name + ".img_acc_enabled", view.img_acc_slice_enabled);
        view.img_acc_slice_level = ptree.get<ushort>(name + ".img_acc_value", view.img_acc_slice_level);
        view.contrast_enabled = ptree.get<bool>(name + ".contrast_enabled", view.contrast_enabled);
        view.contrast_auto_refresh = ptree.get<bool>(name + ".auto_contrast_enabled", view.contrast_auto_refresh);
        view.contrast_invert = ptree.get<bool>(name + ".invert_enabled", view.contrast_invert);
        view.contrast_min_slice = ptree.get<float>(name + ".contrast_min", view.contrast_min_slice);
        view.contrast_max_slice = ptree.get<float>(name + ".contrast_max", view.contrast_max_slice);
    };
    xyz_get("xy", cd.xy);
    xyz_get("xz", cd.xz);
    xyz_get("yz", cd.yz);

    //// Composite
    // cd.composite_kind =
    //     static_cast<CompositeKind>(ptree.get<int>("composite.mode", static_cast<int>(cd.composite_kind.load())));
    cd.composite_auto_weights = ptree.get<bool>("composite.auto_weights_enabled", cd.composite_auto_weights);
    // RGB
    cd.rgb.p_max = ptree.get<ushort>("rgb.p_min", cd.rgb.p_max);
    cd.rgb.p_min = ptree.get<ushort>("rgb.p_max", cd.rgb.p_min);
    cd.rgb.weight_r = ptree.get<float>("rgb.weight_r", cd.rgb.weight_r);
    cd.rgb.weight_g = ptree.get<float>("rgb.weight_g", cd.rgb.weight_g);
    cd.rgb.weight_b = ptree.get<float>("rgb.weight_b", cd.rgb.weight_b);
    // HSV_H
    cd.hsv.h.p_min = ptree.get<ushort>("hsv_h.p_min", cd.hsv.h.p_min);
    cd.hsv.h.p_max = ptree.get<ushort>("hsv_h.p_max", cd.hsv.h.p_max);
    cd.hsv.h.slider_threshold_min = ptree.get<float>("hsv_h.min_value", cd.hsv.h.slider_threshold_min);
    cd.hsv.h.slider_threshold_max = ptree.get<float>("hsv_h.min_value", cd.hsv.h.slider_threshold_max);
    cd.hsv.h.low_threshold = ptree.get<float>("hsv_h.low_threshold", cd.hsv.h.low_threshold);
    cd.hsv.h.high_threshold = ptree.get<float>("hsv_h.high_threshold", cd.hsv.h.high_threshold);
    cd.hsv.h.blur_enabled = ptree.get<bool>("hsv_h.blur_enabled", cd.hsv.h.blur_enabled);
    cd.hsv.h.blur_kernel_size = ptree.get<uint>("hsv_h.blur_size", cd.hsv.h.blur_kernel_size);
    // HSV_S
    cd.hsv.s.p_activated = ptree.get<bool>("hsv_s.enabled", cd.hsv.s.p_activated);
    cd.hsv.s.p_min = ptree.get<ushort>("hsv_s.p_min", cd.hsv.s.p_min);
    cd.hsv.s.p_max = ptree.get<ushort>("hsv_s.p_max", cd.hsv.s.p_max);
    cd.hsv.s.slider_threshold_min = ptree.get<float>("hsv_s.min_value", cd.hsv.s.slider_threshold_min);
    cd.hsv.s.slider_threshold_max = ptree.get<float>("hsv_s.min_value", cd.hsv.s.slider_threshold_max);
    cd.hsv.s.low_threshold = ptree.get<float>("hsv_s.low_threshold", cd.hsv.s.low_threshold);
    cd.hsv.s.high_threshold = ptree.get<float>("hsv_s.high_threshold", cd.hsv.s.high_threshold);
    // HSV_V
    cd.hsv.v.p_activated = ptree.get<bool>("hsv_v.enabled", cd.hsv.v.p_activated);
    cd.hsv.v.p_min = ptree.get<ushort>("hsv_v.p_min", cd.hsv.v.p_min);
    cd.hsv.v.p_max = ptree.get<ushort>("hsv_v.p_max", cd.hsv.v.p_max);
    cd.hsv.v.slider_threshold_min = ptree.get<float>("hsv_v.min_value", cd.hsv.v.slider_threshold_min);
    cd.hsv.v.slider_threshold_max = ptree.get<float>("hsv_v.min_value", cd.hsv.v.slider_threshold_max);
    cd.hsv.v.low_threshold = ptree.get<float>("hsv_v.low_threshold", cd.hsv.v.low_threshold);
    cd.hsv.v.high_threshold = ptree.get<float>("hsv_v.high_threshold", cd.hsv.v.high_threshold);

    // Advanced
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

    // CHECKS AFTER IMPORT
    if (cd.filter2d_n1 >= cd.filter2d_n2)
        cd.filter2d_n1 = cd.filter2d_n2 - 1;
    if (cd.time_transformation_size < 1)
        cd.time_transformation_size = 1;
    // TODO: Check convolution type if it  exists (when it will be added to cd)
    if (cd.p.index >= cd.time_transformation_size)
        cd.p.index = 0;
    if (cd.q.index >= cd.time_transformation_size)
        cd.q.index = 0;
    if (cd.cuts_contrast_p_offset > cd.time_transformation_size - 1)
        cd.cuts_contrast_p_offset = cd.time_transformation_size - 1;

    // Views function call
}

void load_ini(ComputeDescriptor& cd, const std::string& ini_path)
{
    LOG_INFO << "Compute settings loaded from : " << ini_path;

    boost::property_tree::ptree ptree;
    boost::property_tree::ini_parser::read_ini(ini_path, ptree);
    load_ini(ptree, cd);
}

void save_ini(const ComputeDescriptor& cd, const std::string& ini_path)
{
    boost::property_tree::ptree ptree;

    //// Image rendering
    ptree.put<int>("image_rendering.image_mode", static_cast<int>(cd.compute_mode.load()));
    ptree.put<ushort>("image_rendering.batch_size", cd.batch_size);
    ptree.put<ushort>("image_rendering.time_transformation_stride", cd.time_transformation_stride);
    ptree.put<bool>("image_rendering.filter2d_enabled", static_cast<int>(cd.filter2d_enabled.load()));
    ptree.put<int>("image_rendering.filter2d_n1", cd.filter2d_n1.load());
    ptree.put<int>("image_rendering.filter2d_n2", cd.filter2d_n2.load());
    ptree.put<int>("image_rendering.space_transformation", static_cast<int>(cd.space_transformation.load()));
    ptree.put<int>("image_rendering.time_transformation", static_cast<int>(cd.time_transformation.load()));
    ptree.put<ushort>("image_rendering.time_transformation_size", cd.time_transformation_size);
    ptree.put<float>("image_rendering.lambda", cd.lambda);
    ptree.put<float>("image_rendering.z_distance", cd.zdistance);
    ptree.put<bool>("image_rendering.convolution_enabled", cd.convolution_enabled);
    // ptree.put<string>("image_rendering.convolution_type", cd.convolution_type);
    ptree.put<bool>("image_rendering.divide_convolution_enabled", cd.divide_convolution_enabled);

    //// View
    ptree.put<int>("view.view_type", static_cast<int>(cd.img_type.load()));
    // ptree.put<bool>("view.unwrap_2d_enabled", cd.unwrap_2d);
    ptree.put<bool>("view.3d_cuts_enabled", cd.time_transformation_cuts_enabled);
    ptree.put<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);
    ptree.put<bool>("view.lens_view_enabled", cd.lens_view_enabled);
    ptree.put<bool>("view.raw_view_enabled", cd.raw_view_enabled);

    auto xyqp_put = [&](const std::string& name, const AccView& view) {
        ptree.put<bool>("view." + name + "_accu_enabled", view.accu_enabled);
        ptree.put<short>("view." + name + "_accu_level", view.accu_level);
    };
    ptree.put<ushort>("view.x_cuts", cd.x.cuts);
    xyqp_put("x", cd.x);
    ptree.put<ushort>("view.y_cuts", cd.y.cuts);
    xyqp_put("y", cd.y);
    ptree.put<ushort>("view.p_index", cd.p.index);
    xyqp_put("p", cd.p);
    ptree.put<ushort>("view.q_index", cd.q.index);
    xyqp_put("q", cd.q);

    ptree.put<bool>("view.renorm_enabled", cd.renorm_enabled);
    ptree.put<bool>("view.reticle_view_enabled", cd.reticle_view_enabled);
    ptree.put<float>("view.reticle_scale", cd.reticle_scale);

    auto xyz_put = [&](std::string name, const XY_XZ_YZ_WindowView& view) {
        ptree.put<bool>(name + ".flip_enabled", view.flip_enabled);
        ptree.put<int>(name + ".rot", view.rot);
        ptree.put<bool>(name + ".log_scale_enabled", view.log_scale_slice_enabled);
        ptree.put<bool>(name + ".img_acc_enabled", view.img_acc_slice_enabled);
        ptree.put<ushort>(name + ".img_acc_value", view.img_acc_slice_level);
        ptree.put<bool>(name + ".contrast_enabled", view.contrast_enabled);
        ptree.put<bool>(name + ".contrast_auto_enabled", view.contrast_auto_refresh);
        ptree.put<bool>(name + ".contrast_invert_enabled", view.contrast_invert);
        ptree.put<float>(name + ".contrast_min", view.contrast_min_slice);
        ptree.put<float>(name + ".contrast_max", view.contrast_max_slice);
    };
    xyz_put("xy", cd.xy);
    xyz_put("xz", cd.xz);
    xyz_put("yz", cd.yz);

    //// Composite
    ptree.put<int>("composite.mode", static_cast<int>(cd.composite_kind.load()));
    ptree.put<bool>("composite.auto_weights_enabled", cd.composite_auto_weights);
    // RGB
    ptree.put<ushort>("rgb.p_min", cd.rgb.p_min);
    ptree.put<ushort>("rgb.p_max", cd.rgb.p_max);
    ptree.put<float>("rgb.weight_r", cd.rgb.weight_r);
    ptree.put<float>("rgb.weight_g", cd.rgb.weight_g);
    ptree.put<float>("rgb.weight_b", cd.rgb.weight_b);
    // HSV_H
    ptree.put<ushort>("hsv_h.p_min", cd.hsv.h.p_min);
    ptree.put<ushort>("hsv_h.p_max", cd.hsv.h.p_max);
    ptree.put<float>("hsv_h.min_value", cd.hsv.h.slider_threshold_min);
    ptree.put<float>("hsv_h.max_value", cd.hsv.h.slider_threshold_max);
    ptree.put<float>("hsv_h.low_threshold", cd.hsv.h.low_threshold);
    ptree.put<float>("hsv_h.high_threshold", cd.hsv.h.high_threshold);
    ptree.put<bool>("hsv_h.blur_enabled", cd.hsv.h.blur_enabled);
    ptree.put<ushort>("hsv_h.blur_size", cd.hsv.h.blur_kernel_size);
    // HSV_S
    ptree.put<bool>("hsv_s.p_enabled", cd.hsv.s.p_activated);
    ptree.put<ushort>("hsv_s.p_min", cd.hsv.s.p_min);
    ptree.put<ushort>("hsv_s.p_max", cd.hsv.s.p_max);
    ptree.put<float>("hsv_s.min_value", cd.hsv.s.slider_threshold_min);
    ptree.put<float>("hsv_s.max_value", cd.hsv.s.slider_threshold_max);
    ptree.put<float>("hsv_s.low_threshold", cd.hsv.s.low_threshold);
    ptree.put<float>("hsv_s.high_threshold", cd.hsv.s.high_threshold);
    // HSV_V
    ptree.put<bool>("hsv_v.p_enabled", cd.hsv.v.p_activated);
    ptree.put<ushort>("hsv_v.p_min", cd.hsv.v.p_min);
    ptree.put<ushort>("hsv_v.p_max", cd.hsv.v.p_max);
    ptree.put<float>("hsv_v.min_value", cd.hsv.v.slider_threshold_min);
    ptree.put<float>("hsv_v.max_value", cd.hsv.v.slider_threshold_max);
    ptree.put<float>("hsv_v.low_threshold", cd.hsv.v.low_threshold);
    ptree.put<float>("hsv_v.high_threshold", cd.hsv.v.high_threshold);

    // Advanced
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

    boost::property_tree::write_ini(ini_path, ptree);

    LOG_INFO << "Compute settings overwritten at : " << ini_path;
}
} // namespace holovibes::ini
