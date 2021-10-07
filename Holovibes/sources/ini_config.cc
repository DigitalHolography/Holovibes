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
    Config& config = global::global_config;
    // Config
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
    cd.display_rate = ptree.get<float>("config.display_rate", cd.display_rate);

    // Renormalize
    cd.renorm_enabled = ptree.get<bool>("view.renorm_enabled", cd.renorm_enabled);
    cd.renorm_constant = ptree.get<uint>("view.renorm_constant", cd.renorm_constant);

    // Image rendering
    cd.batch_size = ptree.get<ushort>("image_rendering.batch_size", cd.batch_size);

    cd.filter2d_n2 = ptree.get<int>("image_rendering.filter2d_n2", cd.filter2d_n2);
    cd.filter2d_n1 = ptree.get<int>("image_rendering.filter2d_n1", cd.filter2d_n1);
    if (cd.filter2d_n1 >= cd.filter2d_n2)
        cd.filter2d_n1 = cd.filter2d_n2 - 1;
    cd.filter2d_smooth_low = ptree.get<int>("image_rendering.filter2d_smooth_low", cd.filter2d_smooth_low);
    cd.filter2d_smooth_high = ptree.get<int>("image_rendering.filter2d_smooth_high", cd.filter2d_smooth_high);
    cd.filter2d_enabled = ptree.get<bool>("image_rendering.filter2d_enabled", cd.filter2d_enabled);

    const ushort p_time_transformation_size =
        ptree.get<ushort>("image_rendering.time_transformation_size", cd.time_transformation_size);
    if (p_time_transformation_size < 1)
        cd.time_transformation_size = 1;
    else
        cd.time_transformation_size = p_time_transformation_size;

    cd.lambda = ptree.get<float>("image_rendering.lambda", cd.lambda);

    cd.zdistance = ptree.get<float>("image_rendering.z_distance", cd.zdistance);

    cd.space_transformation = static_cast<SpaceTransformation>(
        ptree.get<int>("image_rendering.space_transformation", static_cast<int>(cd.space_transformation.load())));
    cd.time_transformation = static_cast<TimeTransformation>(
        ptree.get<int>("image_rendering.time_transformation", static_cast<int>(cd.time_transformation.load())));

    cd.raw_bitshift = ptree.get<ushort>("image_rendering.raw_bitshift", cd.raw_bitshift);

    cd.time_transformation_stride =
        ptree.get<int>("image_rendering.time_transformation_stride", cd.time_transformation_stride);

    // View
    cd.img_type.exchange(static_cast<ImgType>(ptree.get<int>("view.view_mode", static_cast<int>(cd.img_type.load()))));

    // Displaying mode
    cd.fft_shift_enabled = ptree.get<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);

    cd.log_scale_slice_xy_enabled = ptree.get<bool>("view.log_scale_enabled", cd.log_scale_slice_xy_enabled);
    cd.log_scale_slice_xz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_xz", cd.log_scale_slice_xz_enabled);
    cd.log_scale_slice_yz_enabled = ptree.get<bool>("view.log_scale_enabled_cut_yz", cd.log_scale_slice_yz_enabled);

    const ushort p_index = ptree.get<ushort>("view.p_index", cd.pindex);
    if (p_index >= 0 && p_index < cd.time_transformation_size)
        cd.pindex = p_index;
    const ushort q_index = ptree.get<ushort>("view.q_index", cd.q_index);
    if (q_index >= 0 && q_index < cd.time_transformation_size)
        cd.q_index = q_index;
    const ushort x_cuts = ptree.get<ushort>("view.x_cuts", cd.x_cuts);
    cd.set_x_cuts(x_cuts);
    const ushort y_cuts = ptree.get<ushort>("view.y_cuts", cd.y_cuts);
    cd.set_y_cuts(y_cuts);

    cd.p_accu_enabled = ptree.get<bool>("view.p_accu_enabled", cd.p_accu_enabled);
    cd.q_acc_enabled = ptree.get<bool>("view.q_acc_enabled", cd.q_acc_enabled);
    cd.x_accu_enabled = ptree.get<bool>("view.x_accu_enabled", cd.x_accu_enabled);
    cd.y_accu_enabled = ptree.get<bool>("view.y_accu_enabled", cd.y_accu_enabled);
    cd.p_acc_level = ptree.get<short>("view.p_acc_level", cd.p_acc_level);
    cd.q_acc_level = ptree.get<short>("view.q_acc_level", cd.q_acc_level);
    cd.x_acc_level = ptree.get<short>("view.x_acc_level", cd.x_acc_level);
    cd.y_acc_level = ptree.get<short>("view.y_acc_level", cd.y_acc_level);

    cd.contrast_enabled = ptree.get<bool>("view.contrast_enabled", cd.contrast_enabled);
    cd.contrast_auto_refresh = ptree.get<bool>("view.contrast_auto_refresh", cd.contrast_auto_refresh);
    cd.contrast_invert = ptree.get<bool>("view.contrast_invert", cd.contrast_invert);
    cd.contrast_lower_threshold = ptree.get<float>("view.contrast_lower_threshold", cd.contrast_lower_threshold);
    cd.contrast_upper_threshold = ptree.get<float>("view.contrast_upper_threshold", cd.contrast_upper_threshold);

    cd.contrast_min_slice_xy = ptree.get<float>("view.contrast_min", cd.contrast_min_slice_xy);
    cd.contrast_max_slice_xy = ptree.get<float>("view.contrast_max", cd.contrast_max_slice_xy);
    cd.cuts_contrast_p_offset = ptree.get<ushort>("view.cuts_contrast_p_offset", cd.cuts_contrast_p_offset);
    if (cd.cuts_contrast_p_offset < 0)
        cd.cuts_contrast_p_offset = 0;
    else if (cd.cuts_contrast_p_offset > cd.time_transformation_size - 1)
        cd.cuts_contrast_p_offset = cd.time_transformation_size - 1;

    cd.img_acc_slice_xy_enabled = ptree.get<bool>("view.accumulation_enabled", cd.img_acc_slice_xy_enabled);

    cd.reticle_scale = ptree.get<float>("view.reticle_scale", 0.5f);

    // Import
    cd.pixel_size = ptree.get<float>("import.pixel_size", cd.pixel_size);

    // Composite
    cd.composite_p_red = ptree.get<ushort>("composite.p_red", 1);
    cd.composite_p_blue = ptree.get<ushort>("composite.p_blue", 1);
    cd.weight_r = ptree.get<float>("composite.weight_r", 1);
    cd.weight_g = ptree.get<float>("composite.weight_g", 1);
    cd.weight_b = ptree.get<float>("composite.weight_b", 1);

    cd.composite_p_min_h = ptree.get<ushort>("composite.p_min_h", 1);
    cd.composite_p_max_h = ptree.get<ushort>("composite.p_max_h", 1);
    cd.slider_h_threshold_min = ptree.get<float>("composite.slider_h_threshold_min", 0);
    cd.slider_h_threshold_max = ptree.get<float>("composite.slider_h_threshold_max", 1.0f);
    cd.composite_low_h_threshold = ptree.get<float>("composite.low_h_threshold", 0.2f);
    cd.composite_high_h_threshold = ptree.get<float>("composite.high_h_threshold", 99.8f);

    cd.composite_p_activated_s = ptree.get<bool>("composite.p_activated_s", false);
    cd.composite_p_min_s = ptree.get<ushort>("composite.p_min_s", 1);
    cd.composite_p_max_s = ptree.get<ushort>("composite.p_max_s", 1);
    cd.slider_s_threshold_min = ptree.get<float>("composite.slider_s_threshold_min", 0);
    cd.slider_s_threshold_max = ptree.get<float>("composite.slider_s_threshold_max", 1.0f);
    cd.composite_low_s_threshold = ptree.get<float>("composite.low_s_threshold", 0.2f);
    cd.composite_high_s_threshold = ptree.get<float>("composite.high_s_threshold", 99.8f);

    cd.composite_p_activated_v = ptree.get<bool>("composite.p_activated_v", false);
    cd.composite_p_min_v = ptree.get<ushort>("composite.p_min_v", 1);
    cd.composite_p_max_v = ptree.get<ushort>("composite.p_max_v", 1);
    cd.slider_v_threshold_min = ptree.get<float>("composite.slider_v_threshold_min", 0);
    cd.slider_v_threshold_max = ptree.get<float>("composite.slider_v_threshold_max", 1.0f);
    cd.composite_low_v_threshold = ptree.get<float>("composite.low_v_threshold", 0.2f);
    cd.composite_high_v_threshold = ptree.get<float>("composite.high_v_threshold", 99.8f);

    cd.composite_auto_weights_ = ptree.get<bool>("composite.auto_weights", false);
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
    ptree.put<bool>("view.fft_shift_enabled", cd.fft_shift_enabled);

    ptree.put<int>("view.view_mode", static_cast<int>(cd.img_type.load()));
    ptree.put<bool>("view.log_scale_enabled", cd.log_scale_slice_xy_enabled);
    ptree.put<bool>("view.log_scale_enabled_cut_xz", cd.log_scale_slice_xz_enabled);
    ptree.put<bool>("view.log_scale_enabled_cut_yz", cd.log_scale_slice_yz_enabled);

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

    ptree.put<bool>("view.contrast_enabled", cd.contrast_enabled);
    ptree.put<bool>("view.contrast_auto_refresh", cd.contrast_auto_refresh);
    ptree.put<bool>("view.contrast_invert", cd.contrast_invert);
    ptree.put<float>("view.contrast_lower_threshold", cd.contrast_lower_threshold);
    ptree.put<float>("view.contrast_upper_threshold", cd.contrast_upper_threshold);
    ptree.put<float>("view.contrast_min", cd.contrast_min_slice_xy);
    ptree.put<float>("view.contrast_max", cd.contrast_max_slice_xy);
    ptree.put<ushort>("view.cuts_contrast_p_offset", cd.cuts_contrast_p_offset);
    ptree.put<bool>("view.accumulation_enabled", cd.img_acc_slice_xy_enabled);
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
