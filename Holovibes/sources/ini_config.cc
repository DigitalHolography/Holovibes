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
    LOG_INFO << static_cast<int>(cd.img_type.load());
    // Add unwrap_2d
    cd.time_transformation_cuts_enabled =
        ptree.get<bool>("view.time_transformation_cuts", cd.time_transformation_cuts_enabled);
    cd.fft_shift_enabled = ptree.get<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);
    cd.lens_view_enabled = ptree.get<bool>("view.lens_view_enabled", cd.lens_view_enabled);
    cd.raw_view_enabled = ptree.get<bool>("view.raw_view_enabled", cd.raw_view_enabled);
    // TODO: Create structs and replace 4 * 4 lines by 4 lines
    cd.x_cuts = ptree.get<ushort>("view.x_cuts", cd.x_cuts);
    cd.x_accu_enabled = ptree.get<bool>("view.x_accu_enabled", cd.x_accu_enabled);
    cd.x_acc_level = ptree.get<short>("view.x_acc_level", cd.x_acc_level);
    cd.y_cuts = ptree.get<ushort>("view.y_cuts", cd.y_cuts);
    cd.y_accu_enabled = ptree.get<bool>("view.y_accu_enabled", cd.y_accu_enabled);
    cd.y_acc_level = ptree.get<short>("view.y_acc_level", cd.y_acc_level);
    cd.p_index = ptree.get<ushort>("view.p_index", cd.p_index);
    cd.p_accu_enabled = ptree.get<bool>("view.p_accu_enabled", cd.p_accu_enabled);
    cd.p_acc_level = ptree.get<short>("view.p_acc_level", cd.p_acc_level);
    cd.q_index = ptree.get<ushort>("view.q_index", cd.q_index);
    cd.q_acc_enabled = ptree.get<bool>("view.q_acc_enabled", cd.q_acc_enabled);
    cd.q_acc_level = ptree.get<short>("view.q_acc_level", cd.q_acc_level);
    cd.renorm_enabled = ptree.get<bool>("view.renorm_enabled", cd.renorm_enabled);
    cd.reticle_view_enabled = ptree.get<bool>("view.reticle_view_enabled", cd.p_accu_enabled);
    cd.reticle_scale = ptree.get<float>("view.reticle_scale", cd.reticle_scale);

    // TODO: Struct and 3 function call instead of 3 * 10 lines
    // xy
    cd.xy_flip_enabled = ptree.get<bool>("xy.flip", cd.xy_flip_enabled);
    cd.xy_rot = ptree.get<float>("xy.rot", cd.xy_rot);
    cd.log_scale_slice_xy_enabled = ptree.get<bool>("xy.log_enabled", cd.log_scale_slice_xy_enabled);
    cd.img_acc_slice_xy_enabled = ptree.get<bool>("xy.img_acc_enabled", cd.img_acc_slice_xy_enabled);
    cd.img_acc_slice_xy_level = ptree.get<ushort>("xy.img_acc_value", cd.img_acc_slice_xy_level);
    cd.contrast_enabled = ptree.get<bool>("xy.contrast_enabled", cd.contrast_enabled);
    cd.contrast_auto_refresh = ptree.get<bool>("xy.auto_contrast_enabled", cd.contrast_auto_refresh);
    cd.contrast_invert = ptree.get<bool>("xy.invert_enabled", cd.contrast_invert);
    cd.contrast_min_slice_xy = ptree.get<float>("xy.range_min", cd.contrast_min_slice_xy);
    cd.contrast_max_slice_xy = ptree.get<float>("xy.range_max", cd.contrast_max_slice_xy);
    // xz
    cd.xz_flip_enabled = ptree.get<bool>("xz.flip", cd.xz_flip_enabled);
    cd.xz_rot = ptree.get<float>("xz.rot", cd.xz_rot);
    cd.log_scale_slice_xz_enabled = ptree.get<bool>("xz.log_enabled", cd.log_scale_slice_xz_enabled);
    cd.img_acc_slice_xz_enabled = ptree.get<bool>("xz.img_acc_enabled", cd.img_acc_slice_xz_enabled);
    cd.img_acc_slice_xz_level = ptree.get<ushort>("xz.img_acc_value", cd.img_acc_slice_xz_level);
    cd.xz_contrast_enabled = ptree.get<bool>("xz.contrast_enabled", cd.xz_contrast_enabled);
    cd.xz_contrast_auto_refresh = ptree.get<bool>("xz.auto_contrast_enabled", cd.xz_contrast_auto_refresh);
    cd.xz_contrast_invert = ptree.get<bool>("xz.invert_enabled", cd.xz_contrast_invert);
    cd.contrast_min_slice_xz = ptree.get<float>("xz.range_min", cd.contrast_min_slice_xz);
    cd.contrast_max_slice_xz = ptree.get<float>("xz.range_max", cd.contrast_max_slice_xz);
    // yz
    cd.yz_flip_enabled = ptree.get<bool>("yz.flip", cd.yz_flip_enabled);
    cd.yz_rot = ptree.get<float>("yz.rot", cd.yz_rot);
    cd.log_scale_slice_yz_enabled = ptree.get<bool>("yz.log_enabled", cd.log_scale_slice_yz_enabled);
    cd.img_acc_slice_yz_enabled = ptree.get<bool>("yz.img_acc_enabled", cd.img_acc_slice_yz_enabled);
    cd.img_acc_slice_yz_level = ptree.get<ushort>("yz.img_acc_value", cd.img_acc_slice_yz_level);
    cd.yz_contrast_enabled = ptree.get<bool>("yz.contrast_enabled", cd.yz_contrast_enabled);
    cd.yz_contrast_auto_refresh = ptree.get<bool>("yz.auto_contrast_enabled", cd.yz_contrast_auto_refresh);
    cd.yz_contrast_invert = ptree.get<bool>("yz.invert_enabled", cd.yz_contrast_invert);
    cd.contrast_min_slice_yz = ptree.get<float>("yz.range_min", cd.contrast_min_slice_yz);
    cd.contrast_max_slice_yz = ptree.get<float>("yz.range_max", cd.contrast_max_slice_yz);

    // Composite
    // cd.composite_kind =
    //     static_cast<CompositeKind>(ptree.get<int>("composite.mode", static_cast<int>(cd.composite_kind.load())));
    cd.composite_auto_weights = ptree.get<bool>("composite.auto_weights_enabled", cd.composite_auto_weights);
    // HSV_H
    cd.composite_p_min_h = ptree.get<ushort>("hsv_h.p_min", cd.composite_p_min_h);
    cd.composite_p_max_h = ptree.get<ushort>("hsv_h.p_max", cd.composite_p_max_h);
    cd.composite_slider_h_threshold_min = ptree.get<float>("hsv_h.min_value", cd.composite_slider_h_threshold_min);
    cd.composite_slider_h_threshold_max = ptree.get<float>("hsv_h.min_value", cd.composite_slider_h_threshold_max);
    cd.composite_low_h_threshold = ptree.get<float>("hsv_h.low_threshold", cd.composite_low_h_threshold);
    cd.composite_high_h_threshold = ptree.get<float>("hsv_h.high_threshold", cd.composite_high_h_threshold);
    cd.h_blur_activated = ptree.get<bool>("hsv_h.blur_enabled", cd.h_blur_activated);
    cd.h_blur_activated = ptree.get<uint>("hsv_h.blur_size", cd.h_blur_activated);
    // HSV_S
    cd.composite_p_activated_s = ptree.get<bool>("hsv_s.enabled", cd.composite_p_activated_s);
    cd.composite_p_min_s = ptree.get<ushort>("hsv_s.p_min", cd.composite_p_min_s);
    cd.composite_p_max_s = ptree.get<ushort>("hsv_s.p_max", cd.composite_p_max_s);
    cd.composite_slider_s_threshold_min = ptree.get<float>("hsv_s.min_value", cd.composite_slider_s_threshold_min);
    cd.composite_slider_s_threshold_max = ptree.get<float>("hsv_s.min_value", cd.composite_slider_s_threshold_max);
    cd.composite_low_s_threshold = ptree.get<float>("hsv_s.low_threshold", cd.composite_low_s_threshold);
    cd.composite_high_s_threshold = ptree.get<float>("hsv_s.high_threshold", cd.composite_high_h_threshold);
    // HSV_V
    cd.composite_p_activated_v = ptree.get<bool>("hsv_v.enabled", cd.composite_p_activated_v);
    cd.composite_p_min_v = ptree.get<ushort>("hsv_v.p_min", cd.composite_p_min_v);
    cd.composite_p_max_v = ptree.get<ushort>("hsv_v.p_max", cd.composite_p_max_v);
    cd.composite_slider_v_threshold_min = ptree.get<float>("hsv_v.min_value", cd.composite_slider_v_threshold_min);
    cd.composite_slider_v_threshold_max = ptree.get<float>("hsv_v.min_value", cd.composite_slider_v_threshold_max);
    cd.composite_low_v_threshold = ptree.get<float>("hsv_v.low_threshold", cd.composite_low_v_threshold);
    cd.composite_high_v_threshold = ptree.get<float>("hsv_v.high_threshold", cd.composite_high_h_threshold);

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
    if (cd.p_index >= cd.time_transformation_size)
        cd.p_index = 0;
    if (cd.q_index >= cd.time_transformation_size)
        cd.q_index = 0;
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

    // Image rendering
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

    // View
    ptree.put<int>("view.view_type", static_cast<int>(cd.img_type.load()));
    // ptree.put<bool>("view.unwrap_2d_enabled", cd.unwrap_2d);
    ptree.put<bool>("view.3d_cuts_enabled", cd.time_transformation_cuts_enabled);
    ptree.put<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);
    ptree.put<bool>("view.lens_view_enabled", cd.lens_view_enabled);
    ptree.put<bool>("view.raw_view_enabled", cd.raw_view_enabled);
    ptree.put<ushort>("view.x_cuts", cd.x_cuts);
    ptree.put<bool>("view.x_accu_enabled", cd.x_accu_enabled);
    ptree.put<short>("view.x_acc_level", cd.x_acc_level);
    ptree.put<ushort>("view.y_cuts", cd.y_cuts);
    ptree.put<bool>("view.y_accu_enabled", cd.y_accu_enabled);
    ptree.put<short>("view.y_acc_level", cd.y_acc_level);
    ptree.put<ushort>("view.p_index", cd.p_index);
    ptree.put<bool>("view.p_accu_enabled", cd.p_accu_enabled);
    ptree.put<short>("view.p_acc_level", cd.p_acc_level);
    ptree.put<ushort>("view.q_index", cd.q_index);
    ptree.put<bool>("view.q_accu_enabled", cd.q_acc_enabled);
    ptree.put<short>("view.q_acc_level", cd.q_acc_level);
    ptree.put<bool>("view.renorm_enabled", cd.renorm_enabled);
    ptree.put<bool>("view.reticle_view_enabled", cd.reticle_view_enabled);
    ptree.put<float>("view.reticle_scale", cd.reticle_scale);

    // xy
    ptree.put<bool>("xy.flip", cd.xy_flip_enabled);
    ptree.put<int>("xy.rot", cd.xy_rot);
    ptree.put<bool>("xy.log_enabled", cd.log_scale_slice_xy_enabled);
    ptree.put<bool>("xy.img_acc_enabled", cd.img_acc_slice_xy_enabled);
    ptree.put<ushort>("xy.img_acc_value", cd.img_acc_slice_xy_level);
    ptree.put<bool>("xy.contrast_enabled", cd.contrast_enabled);
    ptree.put<bool>("xy.contrast_auto_enabled", cd.contrast_auto_refresh);
    ptree.put<bool>("xy.contrast_invert_enabled", cd.contrast_invert);
    ptree.put<float>("xy.contrast_min", cd.contrast_min_slice_xy);
    ptree.put<float>("xy.contrast_max", cd.contrast_max_slice_xy);
    // xz
    ptree.put<bool>("xz.flip", cd.xz_flip_enabled);
    ptree.put<int>("xz.rot", cd.xz_rot);
    ptree.put<bool>("xz.log_enabled", cd.log_scale_slice_xz_enabled);
    ptree.put<bool>("xz.img_acc_enabled", cd.img_acc_slice_xz_enabled);
    ptree.put<ushort>("xz.img_acc_value", cd.img_acc_slice_xz_level);
    ptree.put<bool>("xz.contrast_enabled", cd.xz_contrast_enabled);
    ptree.put<bool>("xz.contrast_auto_enabled", cd.xz_contrast_auto_refresh);
    ptree.put<bool>("xz.contrast_invert_enabled", cd.xz_contrast_invert);
    ptree.put<float>("xz.contrast_min", cd.contrast_min_slice_xz);
    ptree.put<float>("xz.contrast_max", cd.contrast_max_slice_xz);
    // yz
    ptree.put<bool>("yz.flip", cd.yz_flip_enabled);
    ptree.put<int>("yz.rot", cd.yz_rot);
    ptree.put<bool>("yz.log_enabled", cd.log_scale_slice_yz_enabled);
    ptree.put<bool>("yz.img_acc_enabled", cd.img_acc_slice_yz_enabled);
    ptree.put<ushort>("yz.img_acc_value", cd.img_acc_slice_yz_level);
    ptree.put<bool>("yz.contrast_enabled", cd.yz_contrast_enabled);
    ptree.put<bool>("yz.contrast_auto_enabled", cd.yz_contrast_auto_refresh);
    ptree.put<bool>("yz.contrast_invert_enabled", cd.yz_contrast_invert);
    ptree.put<float>("yz.contrast_min", cd.contrast_min_slice_yz);
    ptree.put<float>("yz.contrast_max", cd.contrast_max_slice_yz);

    // ptree.put<bool>("view.accumulation_enabled", cd.img_acc_slice_xy_enabled);
    //
    // ptree.put<bool>("view.contrast_enabled", cd.contrast_enabled);
    // ptree.put<bool>("view.contrast_auto_refresh", cd.contrast_auto_refresh);
    // ptree.put<bool>("view.contrast_invert", cd.contrast_invert);
    // ptree.put<float>("view.contrast_min", cd.contrast_min_slice_xy);
    // ptree.put<float>("view.contrast_max", cd.contrast_max_slice_xy);
    //
    // // Composite
    // cd.composite_kind =
    //     static_cast<CompositeKind>(ptree.get<int>("composite.mode", static_cast<int>(cd.composite_kind.load())));
    // ptree.put<bool>("composite.auto_weights_enabled", cd.composite_auto_weights);
    // // RGB
    // ptree.put<ushort>("rgb.p_min", cd.rgb.min);
    // ptree.put<ushort>("rgb.p_max", cd.rgb.max);
    // ptree.put<float>("rgb.weight_r", cd.rgb.r_weight);
    // ptree.put<float>("rgb.weight_g", cd.rgb.g_weight);
    // ptree.put<float>("rgb.weight_b", cd.rgb.b_weight);
    // // HSV
    // ptree.put<bool>("hsv.s_enabled", cd.hsv.s_enabled);
    // ptree.put<bool>("hsv.v_enabled", cd.hsv.v_enabled);
    // // HSV_H
    // ptree.put<ushort>("hsv_h.p_min", cd.hsv.h.p_min);
    // ptree.put<ushort>("hsv_h.p_max", cd.hsv.h.p_max);
    // ptree.put<float>("hsv_h.min_value", cd.hsv.h.min_value);
    // ptree.put<float>("hsv_h.max_value", cd.hsv.h.max_value);
    // ptree.put<bool>("hsv_h.blur_enabled", cd.hsv.h.blur_enabled);
    // ptree.put<ushort>("hsv_h.blur_size", cd.hsv.h.blur_size);
    // ptree.put<float>("hsv_h.low_threshold", cd.hsv.h.low_threshold);
    // ptree.put<float>("hsv_h.high_threshold", cd.hsv.h.high_threshold);
    // // HSV_S
    // ptree.put<ushort>("hsv_s.p_min", cd.hsv.s.p_min);
    // ptree.put<ushort>("hsv_s.p_max", cd.hsv.s.p_max);
    // ptree.put<float>("hsv_s.min_value", cd.hsv.s.min_value);
    // ptree.put<float>("hsv_s.max_value", cd.hsv.s.max_value);
    // ptree.put<float>("hsv_s.low_threshold", cd.hsv.s.low_threshold);
    // ptree.put<float>("hsv_s.high_threshold", cd.hsv.s.high_threshold);
    // // HSV_V
    // ptree.put<ushort>("hsv_v.p_min", cd.hsv.v.p_min);
    // ptree.put<ushort>("hsv_v.p_max", cd.hsv.v.p_max);
    // ptree.put<float>("hsv_v.min_value", cd.hsv.v.min_value);
    // ptree.put<float>("hsv_v.max_value", cd.hsv.v.max_value);
    // ptree.put<float>("hsv_v.low_threshold", cd.hsv.v.low_threshold);
    // ptree.put<float>("hsv_v.high_threshold", cd.hsv.v.high_threshold);

    // Composite
    ptree.put<int>("composite.mode", static_cast<int>(cd.composite_kind.load()));
    ptree.put<bool>("composite.auto_weights_enabled", cd.composite_auto_weights);
    // RGB
    ptree.put<ushort>("rgb.p_min", cd.rgb_p_min);
    ptree.put<ushort>("rgb.p_max", cd.rgb_p_max);
    ptree.put<float>("rgb.weight_r", cd.weight_r);
    ptree.put<float>("rgb.weight_g", cd.weight_g);
    ptree.put<float>("rgb.weight_b", cd.weight_b);
    // HSV_H
    ptree.put<ushort>("hsv_h.p_min", cd.composite_p_min_h);
    ptree.put<ushort>("hsv_h.p_max", cd.composite_p_max_h);
    ptree.put<float>("hsv_h.min_value", cd.composite_slider_h_threshold_min);
    ptree.put<float>("hsv_h.max_value", cd.composite_slider_h_threshold_max);
    ptree.put<float>("hsv_h.low_threshold", cd.composite_low_h_threshold);
    ptree.put<float>("hsv_h.high_threshold", cd.composite_high_h_threshold);
    ptree.put<bool>("hsv_h.blur_enabled", cd.h_blur_activated);
    ptree.put<ushort>("hsv_h.blur_size", cd.h_blur_kernel_size);
    // HSV_S
    ptree.put<bool>("hsv_s.p_enabled", cd.composite_p_activated_s);
    ptree.put<ushort>("hsv_s.p_min", cd.composite_p_min_s);
    ptree.put<ushort>("hsv_s.p_max", cd.composite_p_max_s);
    ptree.put<float>("hsv_s.min_value", cd.composite_slider_s_threshold_min);
    ptree.put<float>("hsv_s.max_value", cd.composite_slider_s_threshold_max);
    ptree.put<float>("hsv_s.low_threshold", cd.composite_low_s_threshold);
    ptree.put<float>("hsv_s.high_threshold", cd.composite_high_s_threshold);
    // HSV_V
    ptree.put<bool>("hsv_v.p_enabled", cd.composite_p_activated_v);
    ptree.put<ushort>("hsv_v.p_min", cd.composite_p_min_v);
    ptree.put<ushort>("hsv_v.p_max", cd.composite_p_max_v);
    ptree.put<float>("hsv_v.min_value", cd.composite_slider_v_threshold_min);
    ptree.put<float>("hsv_v.max_value", cd.composite_slider_v_threshold_max);
    ptree.put<float>("hsv_v.low_threshold", cd.composite_low_v_threshold);
    ptree.put<float>("hsv_v.high_threshold", cd.composite_high_v_threshold);

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
