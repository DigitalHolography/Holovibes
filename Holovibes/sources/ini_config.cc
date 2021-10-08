#include "ini_config.hh"

namespace holovibes::ini
{

void load_ini(ComputeDescriptor& cd, const std::string& ini_path)
{
    LOG_DEBUG << "Loading ini file at path: " << ini_path;

    boost::property_tree::ptree ptree;
    boost::property_tree::ini_parser::read_ini(ini_path, ptree);
    load_ini(ptree, cd);
}

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
    // to create
    // cd.convolution_type= ptree.get<bool>("image_rendering.convolution_enabled", cd.convolution_enabled);
    cd.divide_convolution_enabled =
        ptree.get<bool>("image_rendering.divide_convolution_enabled", cd.divide_convolution_enabled);

    // View
    cd.img_type.exchange(static_cast<ImgType>(ptree.get<int>("view.view_mode", static_cast<int>(cd.img_type.load()))));
    // Add unwrap_2d
    cd.time_transformation_cuts_enabled =
        ptree.get<bool>("view.time_transformation_cuts", cd.time_transformation_cuts_enabled);
    cd.fft_shift_enabled = ptree.get<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);
    cd.gpu_lens_display_enabled = ptree.get<bool>("view.lens_view_enabled", cd.gpu_lens_display_enabled);
    cd.raw_view_enabled = ptree.get<bool>("view.raw_view_enabled", cd.raw_view_enabled);
    // TODO: Create struct
    cd.x_cuts = ptree.get<ushort>("view.x_cuts", cd.x_cuts);
    cd.x_accu_enabled = ptree.get<bool>("view.x_accu_enabled", cd.x_accu_enabled);
    cd.x_acc_level = ptree.get<short>("view.x_acc_level", cd.x_acc_level);
    cd.y_cuts = ptree.get<ushort>("view.y_cuts", cd.y_cuts);
    cd.y_accu_enabled = ptree.get<bool>("view.y_accu_enabled", cd.y_accu_enabled);
    cd.y_acc_level = ptree.get<short>("view.y_acc_level", cd.y_acc_level);
    cd.pindex = ptree.get<ushort>("view.pindex", cd.pindex);
    cd.p_accu_enabled = ptree.get<bool>("view.p_accu_enabled", cd.p_accu_enabled);
    cd.p_acc_level = ptree.get<short>("view.p_acc_level", cd.p_acc_level);
    cd.q_index = ptree.get<ushort>("view.q_index", cd.q_index);
    cd.q_acc_enabled = ptree.get<bool>("view.q_acc_enabled", cd.q_acc_enabled);
    cd.q_acc_level = ptree.get<short>("view.q_acc_level", cd.q_acc_level);
    cd.renorm_enabled = ptree.get<bool>("view.renorm_enabled", cd.renorm_enabled);
    cd.reticle_enabled = ptree.get<bool>("view.reticle_enabled", cd.p_accu_enabled);
    cd.reticle_scale = ptree.get<float>("view.reticle_scale", cd.reticle_scale);

    // xy => Settings to merge in a struct (some are in MainWindow, others in cd)
    //// cd.xy.flip = ptree.get<bool>("xy.flip", cd.xy.flip);
    //// cd.xy.rot = ptree.get<float>("xy.rot", cd.xy.rot);
    //// cd.xy.log_enabled = ptree.get<bool>("xy.log_enabled", cd.xy.log_enabled);
    //// cd.xy.img_acc_enabled = ptree.get<bool>("xy.img_acc_enabled", cd.xy.img_acc_enabled);
    //// cd.xy.img_acc_value = ptree.get<ushort>("xy.img_acc_value", cd.p_acc_level);
    //// cd.xy.auto_enabled = ptree.get<bool>("xy.auto_contrast_enabled", cd.xy.log_enabled);
    //// cd.xy.invert_enabled = ptree.get<bool>("xy.invert_enabled", cd.xy.invert_enabled);
    //// cd.xy.range_min = ptree.get<float>("xy.range_min", cd.xy.range_min);
    //// cd.xy.range_max = ptree.get<float>("xy.range_max", cd.xy.range_max);
    // xz
    // ==> Duplicate the section xy (changing all 'xy' by 'xz').
    // yz
    // ==> Duplicate the section xy (changing all 'xy' by 'yz').
    // Current for xy / xz / yz
    cd.log_scale_slice_xy_enabled = ptree.get<bool>("view.log_scale_enabled", cd.log_scale_slice_xy_enabled);
    cd.log_scale_slice_xz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_xz", cd.log_scale_slice_xz_enabled);
    cd.log_scale_slice_yz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_yz", cd.log_scale_slice_yz_enabled);
    cd.img_acc_slice_xy_enabled = ptree.get<bool>("view.accumulation_enabled", cd.img_acc_slice_xy_enabled);
    cd.contrast_enabled = ptree.get<bool>("view.contrast_enabled", cd.contrast_enabled);
    cd.contrast_auto_refresh = ptree.get<bool>("view.contrast_auto_refresh", cd.contrast_auto_refresh);
    cd.contrast_invert = ptree.get<bool>("view.contrast_invert", cd.contrast_invert);
    cd.contrast_min_slice_xy = ptree.get<float>("view.contrast_min", cd.contrast_min_slice_xy);
    cd.contrast_max_slice_xy = ptree.get<float>("view.contrast_max", cd.contrast_max_slice_xy);

    // // Composite // TODO WHEN IT WILL BE IMPLEMENTED
    // cd.composite_kind =
    //     static_cast<CompositeKind>(ptree.get<int>("composite.mode", static_cast<int>(cd.composite_kind.load())));
    // cd.composite_auto_weights_ = ptree.get<bool>("composite.auto_weights_enabled", cd.composite_auto_weights_);
    // // RGB
    // cd.rgb.p_min = ptree.get<ushort>("rgb.min", cd.rgb.p_min); // p_red => min
    // cd.rgb.p_max = ptree.get<ushort>("rgb.max", cd.rgb.p_max); // p_blue => max
    // cd.rgb.r_weight = ptree.get<float>("rgb.weight_r", cd.rgb.r_weight);
    // cd.rgb.g_weight = ptree.get<float>("rgb.weight_g", cd.rgb.g_weight);
    // cd.rgb.b_weight = ptree.get<float>("rgb.weight_b", cd.rgb.b_weight);
    // // HSV_H
    // cd.hsv.h.p_min = ptree.get<ushort>("hsv_h.p_min", 1);
    // cd.hsv.h.p_max = ptree.get<ushort>("hsv_h.p_max", 1);
    // cd.hsv.h.min_value = ptree.get<float>("hsv_h.min_value", 0);
    // cd.hsv.h.max_value = ptree.get<float>("hsv_h.max_value", 1.0f);
    // cd.hsv.h.blur_enabled = ptree.get<bool>("hsv_h.blur", cd.hsv.h.blur_enabled); // set_h_blur_kernel_size => blur
    // cd.hsv.h.blur_size = ptree.get<ushort>("hsv_h.blur_size", cd.hsv.h.blur_size);
    // //// The next 2 are surely to remove
    // cd.hsv.h.low_threshold = ptree.get<float>("hsv_h.low_threshold", 0.2f);
    // cd.hsv.h.high_threshold = ptree.get<float>("hsv_h.high_threshold", 99.8f);
    // // HSV_S
    // cd.hsv.s.p_min = ptree.get<ushort>("hsv_s.p_min", 1);
    // cd.hsv.s.p_max = ptree.get<ushort>("hsv_s.p_max", 1);
    // cd.hsv.s.min_value = ptree.get<float>("hsv_s.min_value", 0);
    // cd.hsv.s.max_value = ptree.get<float>("hsv_s.max_value", 1.0f);
    // //// The sext 2 are surely to remove
    // cd.hsv.s.low_threshold = ptree.get<float>("hsv_s.low_threshold", 0.2f);
    // cd.hsv.s.high_threshold = ptree.get<float>("hsv_s.high_threshold", 99.8f);
    // // HSV_V
    // cd.hsv.v.p_min = ptree.get<ushort>("hsv_v.p_min", 1);
    // cd.hsv.v.p_max = ptree.get<ushort>("hsv_v.p_max", 1);
    // cd.hsv.v.min_value = ptree.get<float>("hsv_v.min_value", 0);
    // cd.hsv.v.max_value = ptree.get<float>("hsv_v.max_value", 1.0f);
    // //// The vext 2 are surely to remove
    // cd.hsv.v.low_threshold = ptree.get<float>("hsv_v.low_threshold", 0.2f);
    // cd.hsv.v.high_threshold = ptree.get<float>("hsv_v.high_threshold", 99.8f);

    // Current Composite (to refa with struct) for HSV
    // H
    cd.composite_p_min_h = ptree.get<ushort>("composite.p_min_h", 1);
    cd.composite_p_max_h = ptree.get<ushort>("composite.p_max_h", 1);
    cd.slider_h_threshold_min = ptree.get<float>("hsv_.slider_h_threshold_min", 0);
    cd.slider_h_threshold_max = ptree.get<float>("hsv_.slider_h_threshold_max", 1.0f);
    cd.composite_low_h_threshold = ptree.get<float>("composite.low_h_threshold", 0.2f);
    cd.composite_high_h_threshold = ptree.get<float>("composite.high_h_threshold", 99.8f);
    // S
    cd.composite_p_activated_s = ptree.get<bool>("composite.p_activated_s", false);
    cd.composite_p_min_s = ptree.get<ushort>("composite.p_min_s", 1);
    cd.composite_p_max_s = ptree.get<ushort>("composite.p_max_s", 1);
    cd.slider_s_threshold_min = ptree.get<float>("hsv_.slider_s_threshold_min", 0);
    cd.slider_s_threshold_max = ptree.get<float>("hsv_.slider_s_threshold_max", 1.0f);
    cd.composite_low_s_threshold = ptree.get<float>("composite.low_s_threshold", 0.2f);
    cd.composite_high_s_threshold = ptree.get<float>("composite.high_s_threshold", 99.8f);
    // V
    cd.composite_p_activated_v = ptree.get<bool>("composite.p_activated_v", false);
    cd.composite_p_min_v = ptree.get<ushort>("composite.p_min_v", 1);
    cd.composite_p_max_v = ptree.get<ushort>("composite.p_max_v", 1);
    cd.slider_v_threshold_min = ptree.get<float>("composite.slider_v_threshold_min", 0);
    cd.slider_v_threshold_max = ptree.get<float>("composite.slider_v_threshold_max", 1.0f);
    cd.composite_low_v_threshold = ptree.get<float>("composite.low_v_threshold", 0.2f);
    cd.composite_high_v_threshold = ptree.get<float>("composite.high_v_threshold", 99.8f);
    // end cur

    // NOT ON GUI SETTINGS
    // // Config
    // cd.file_buffer_size = ptree.get<ushort>("not_on_gui_settings.file_buffer_size", cd.file_buffer_size);
    // cd.input_buffer_size = ptree.get<ushort>("not_on_gui_settings.input_buffer_size", cd.input_buffer_size);
    // cd.record_buffer_size = ptree.get<ushort>("not_on_gui_settings.record_buffer_size", cd.record_buffer_size);
    // cd.output_buffer_size = ptree.get<ushort>("not_on_gui_settings.output_buffer_size", cd.output_buffer_size);
    // cd.time_transformation_cuts_output_buffer_size =
    //     ptree.get<ushort>("not_on_gui_settings.time_transformation_cuts_output_buffer_size",
    //                       cd.time_transformation_cuts_output_buffer_size);
    // cd.accumulation_buffer_size = // img_acc_slice_xy_level => accumulation_buffer_size
    //     ptree.get<ushort>("not_on_gui_settings.accumulation_buffer_size", cd.accumulation_buffer_size);
    // cd.frame_timeout = ptree.get<int>("not_on_gui_settings.frame_timeout", cd.frame_timeout);
    Config& config = global::global_config;
    cd.display_rate = ptree.get<float>("not_on_gui_settings.display_rate", cd.display_rate);
    cd.filter2d_smooth_low = ptree.get<int>("not_on_gui_settings.filter2d_smooth_low", cd.filter2d_smooth_low);
    cd.filter2d_smooth_high = ptree.get<int>("not_on_gui_settings.filter2d_smooth_high", cd.filter2d_smooth_high);
    cd.contrast_lower_threshold =
        ptree.get<float>("not_on_gui_settings.contrast_lower_threshold", cd.contrast_lower_threshold);
    cd.contrast_upper_threshold =
        ptree.get<float>("not_on_gui_settings.contrast_upper_threshold", cd.contrast_upper_threshold);
    cd.renorm_constant = ptree.get<uint>("not_on_gui_settings.renorm_constant", cd.renorm_constant);
    cd.cuts_contrast_p_offset = ptree.get<ushort>("view.cuts_contrast_p_offset", cd.cuts_contrast_p_offset);
    // cd.auto_scale_point_threshold_ = // Mainwindow => cd
    //         ptree.get<size_t>("chart.auto_scale_point_threshold", cd.auto_scale_point_threshold_);
    // beg Current config => to move in cd
    config.file_buffer_size = ptree.get<int>("config.file_buffer_size", config.file_buffer_size);
    config.input_queue_max_size = ptree.get<int>("config.input_buffer_size", config.input_queue_max_size);
    config.frame_record_queue_max_size =
        ptree.get<int>("config.record_buffer_size", config.frame_record_queue_max_size);
    config.output_queue_max_size = ptree.get<int>("config.output_buffer_size", config.output_queue_max_size);
    config.time_transformation_cuts_output_buffer_size =
        ptree.get<int>("config.time_transformation_cuts_output_buffer_size",
                       config.time_transformation_cuts_output_buffer_size);
    config.frame_timeout = ptree.get<int>("config.frame_timeout", config.frame_timeout);
    cd.img_acc_slice_xy_level = ptree.get<uint>("config.accumulation_buffer_size", cd.img_acc_slice_xy_level);
    // end cur

    // CHECKS AFTER IMPORT
    if (cd.filter2d_n1 >= cd.filter2d_n2)
        cd.filter2d_n1 = cd.filter2d_n2 - 1;
    if (cd.time_transformation_size < 1)
        cd.time_transformation_size = 1;
    // check convolution type if it  exists
    if (cd.pindex >= cd.time_transformation_size)
        cd.pindex = 0;
    if (cd.q_index >= cd.time_transformation_size)
        cd.q_index = 0;
    if (cd.cuts_contrast_p_offset > cd.time_transformation_size - 1)
        cd.cuts_contrast_p_offset = cd.time_transformation_size - 1;
}

void save_ini(boost::property_tree::ptree& ptree, const ComputeDescriptor& cd)
{
    const Config& config = global::global_config;

    // Config
    ptree.put<uint>("config.file_buffer_size", config.file_buffer_size);
    ptree.put<uint>("config.input_buffer_size", config.input_queue_max_size);
    ptree.put<uint>("config.record_buffer_size", config.frame_record_queue_max_size);
    ptree.put<uint>("config.output_buffer_size", config.output_queue_max_size);
    ptree.put<uint>("config.time_transformation_cuts_output_buffer_size",
                    config.time_transformation_cuts_output_buffer_size);
    ptree.put<uint>("config.accumulation_buffer_size", cd.img_acc_slice_xy_level);
    ptree.put<uint>("config.frame_timeout", config.frame_timeout);
    ptree.put<ushort>("config.display_rate", static_cast<ushort>(cd.display_rate));

    // Image rendering
    ptree.put<int>("image_rendering.image_mode", static_cast<int>(cd.compute_mode.load()));
    ptree.put<ushort>("image_rendering.batch_size", cd.batch_size);
    ptree.put<ushort>("image_rendering.time_transformation_stride", cd.time_transformation_stride);
    ptree.put<bool>("image_rendering.filter2d_enabled", static_cast<int>(cd.filter2d_enabled.load()));
    ptree.put<int>("image_rendering.filter2d_n1", cd.filter2d_n1.load());
    ptree.put<int>("image_rendering.filter2d_n2", cd.filter2d_n2.load());
    ptree.put<int>("image_rendering.filter2d_smooth_low", cd.filter2d_smooth_low.load());
    ptree.put<int>("image_rendering.filter2d_smooth_high", cd.filter2d_smooth_high.load());
    ptree.put<int>("image_rendering.space_transformation", static_cast<int>(cd.space_transformation.load()));
    ptree.put<int>("image_rendering.time_transformation", static_cast<int>(cd.time_transformation.load()));
    ptree.put<ushort>("image_rendering.time_transformation_size", cd.time_transformation_size);
    ptree.put<float>("image_rendering.lambda", cd.lambda);
    ptree.put<float>("image_rendering.z_distance", cd.zdistance);
    ptree.put<ushort>("image_rendering.raw_bitshift", cd.raw_bitshift);

    // View
    ptree.put<int>("view.view_mode", static_cast<int>(cd.img_type.load()));

    ptree.put<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);

    // to delete // ptree.put<bool>("view.log_scale_enabled", cd.log_scale_slice_xy_enabled);
    // to delete // ptree.put<bool>("view.log_scale_enabled_cut_xz", cd.log_scale_slice_xz_enabled);
    // to delete // ptree.put<bool>("view.log_scale_enabled_cut_yz", cd.log_scale_slice_yz_enabled);

    ptree.put<ushort>("view.p_index", cd.pindex);
    ptree.put<ushort>("view.q_index", cd.q_index);
    ptree.put<ushort>("view.x_cuts", cd.x_cuts);
    ptree.put<ushort>("view.y_cuts", cd.y_cuts);
    ptree.put<bool>("view.p_accu_enabled", cd.p_accu_enabled);
    ptree.put<bool>("view.q_accu_enabled", cd.q_acc_enabled);
    ptree.put<bool>("view.x_accu_enabled", cd.x_accu_enabled);
    ptree.put<bool>("view.y_accu_enabled", cd.y_accu_enabled);
    ptree.put<short>("view.p_acc_level", cd.p_acc_level);
    ptree.put<short>("view.q_acc_level", cd.q_acc_level);
    ptree.put<short>("view.x_acc_level", cd.x_acc_level);
    ptree.put<short>("view.y_acc_level", cd.y_acc_level);

    ptree.put<bool>("view.accumulation_enabled", cd.img_acc_slice_xy_enabled);

    ptree.put<bool>("view.contrast_enabled", cd.contrast_enabled);
    ptree.put<bool>("view.contrast_auto_refresh", cd.contrast_auto_refresh);
    ptree.put<bool>("view.contrast_invert", cd.contrast_invert);
    ptree.put<float>("view.contrast_lower_threshold", cd.contrast_lower_threshold);
    ptree.put<float>("view.contrast_upper_threshold", cd.contrast_upper_threshold);
    ptree.put<float>("view.contrast_min", cd.contrast_min_slice_xy);
    ptree.put<float>("view.contrast_max", cd.contrast_max_slice_xy);
    ptree.put<ushort>("view.cuts_contrast_p_offset", cd.cuts_contrast_p_offset);

    ptree.put<float>("view.reticle_scale", cd.reticle_scale);

    ptree.put<bool>("view.renorm_enabled", cd.renorm_enabled);
    ptree.put<uint>("view.renorm_constant", cd.renorm_constant);

    // Import
    ptree.put<float>("import.pixel_size", cd.pixel_size);

    // Composite
    ptree.put<ushort>("composite.p_red", cd.composite_p_red);
    ptree.put<ushort>("composite.p_blue", cd.composite_p_blue);
    ptree.put<float>("composite.weight_r", cd.weight_r);
    ptree.put<float>("composite.weight_g", cd.weight_g);
    ptree.put<float>("composite.weight_b", cd.weight_b);

    ptree.put<ushort>("composite.p_min_h", cd.composite_p_min_h);
    ptree.put<ushort>("composite.p_max_h", cd.composite_p_max_h);
    ptree.put<float>("composite.slider_h_threshold_min", cd.slider_h_threshold_min);
    ptree.put<float>("composite.slider_h_threshold_max", cd.slider_h_threshold_max);
    ptree.put<float>("composite.low_h_threshold", cd.composite_low_h_threshold);
    ptree.put<float>("composite.high_h_threshold", cd.composite_high_h_threshold);

    ptree.put<bool>("composite.p_activated_s", cd.composite_p_activated_s);
    ptree.put<ushort>("composite.p_min_s", cd.composite_p_min_s);
    ptree.put<ushort>("composite.p_max_s", cd.composite_p_max_s);
    ptree.put<float>("composite.slider_s_threshold_min", cd.slider_s_threshold_min);
    ptree.put<float>("composite.slider_s_threshold_max", cd.slider_s_threshold_max);
    ptree.put<float>("composite.low_s_threshold", cd.composite_low_s_threshold);
    ptree.put<float>("composite.high_s_threshold", cd.composite_high_s_threshold);

    ptree.put<bool>("composite.p_activated_v", cd.composite_p_activated_v);
    ptree.put<ushort>("composite.p_min_v", cd.composite_p_min_v);
    ptree.put<ushort>("composite.p_max_v", cd.composite_p_max_v);
    ptree.put<float>("composite.slider_v_threshold_min", cd.slider_v_threshold_min);
    ptree.put<float>("composite.slider_v_threshold_max", cd.slider_v_threshold_max);
    ptree.put<float>("composite.low_v_threshold", cd.composite_low_v_threshold);
    ptree.put<float>("composite.high_v_threshold", cd.composite_high_v_threshold);
    ptree.put<bool>("composite.auto_weights", cd.composite_auto_weights_);
}
} // namespace holovibes::ini
