#include "API.hh"

namespace holovibes::api
{
void load_image_rendering(const json& data)
{
    ComputeDescriptor& cd = api::get_cd();

    cd.compute_mode = string_to_computation[data["image mode"]];
    cd.batch_size = data["batch size"];
    cd.time_transformation_stride = data["time transformation stride"];

    const json& filter_2d_data = data["filter2d"];
    cd.filter2d_enabled = filter_2d_data["enabled"];
    cd.filter2d_n1 = filter_2d_data["n1"];
    cd.filter2d_n2 = filter_2d_data["n2"];

    cd.space_transformation = string_to_space_transformation[data["space transformation"]];
    cd.time_transformation = string_to_time_transformation[data["time transformation"]];
    cd.time_transformation_size = data["time transformation size"];
    cd.lambda = data["lambda"];
    cd.zdistance = data["z distance"];

    const json& convolution_data = data["convolution"];
    cd.convolution_enabled = convolution_data["enabled"];
    // FIXME: Use GSH instead of UID
    UserInterfaceDescriptor::instance().convo_name = convolution_data["type"];
    // FIXME: Loads convolution matrix
    cd.divide_convolution_enabled = convolution_data["divide"];

    // FIXME: Need to use setters that are currently in CD
    // Example, matrix is not loaded if we do not pass through setter
}

void load_view(const json& data)
{
    ComputeDescriptor& cd = api::get_cd();

    cd.img_type = string_to_img_type[data["type"]];
    cd.fft_shift_enabled = data["fft shift"];
    cd.x.from_json(data["x"]);
    cd.y.from_json(data["y"]);
    cd.p.from_json(data["p"]);
    cd.q.from_json(data["q"]);

    const json& window_data = data["window"];
    cd.xy.from_json(window_data["xy"]);
    cd.yz.from_json(window_data["yz"]);
    cd.xz.from_json(window_data["xz"]);
    cd.filter2d.from_json(window_data["filter2d"]);

    cd.renorm_enabled = data["renorm"];

    const json& reticle_data = data["reticle"];
    cd.reticle_display_enabled = reticle_data["display enabled"];
    cd.reticle_scale = reticle_data["scale"];
}

void load_composite(const json& data)
{
    ComputeDescriptor& cd = api::get_cd();
    cd.composite_kind = string_to_composite_kind[data["mode"]];
    cd.composite_auto_weights = data["auto weight"];
    cd.rgb.from_json(data["rgb"]);
    cd.hsv.from_json(data["hsv"]);
}
void load_advanced(const json& data)
{
    ComputeDescriptor& cd = api::get_cd();

    const json& buffer_size_data = data["buffer size"];
    cd.file_buffer_size = buffer_size_data["file"];
    cd.input_buffer_size = buffer_size_data["input"];
    cd.output_buffer_size = buffer_size_data["output"];
    cd.record_buffer_size = buffer_size_data["record"];
    cd.time_transformation_cuts_output_buffer_size = buffer_size_data["time transformation cuts"];

    const json& contrast_data = data["contrast"];
    cd.contrast_lower_threshold = contrast_data["lower"];
    cd.contrast_upper_threshold = contrast_data["upper"];
    cd.cuts_contrast_p_offset = contrast_data["cuts p offset"];

    const json& filter2d_smooth_data = data["filter2d smooth"];
    cd.filter2d_smooth_high = filter2d_smooth_data["high"];
    cd.filter2d_smooth_low = filter2d_smooth_data["low"];

    cd.renorm_constant = data["renorm constant"];
}

void json_to_compute_settings(const json& data)
{
    load_image_rendering(data["image rendering"]);
    load_view(data["view"]);
    load_composite(data["composite"]);
    load_advanced(data["advanced"]);
}

void after_load_checks(ComputeDescriptor& cd)
{
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
}

void load_compute_settings(const std::string& json_path)
{
    if (json_path.empty())
        return;

    LOG_INFO << "Compute settings loaded from : " << json_path;

    std::ifstream ifs(json_path);
    auto j_cs = json::parse(ifs);

    json_to_compute_settings(j_cs);

    after_load_checks(get_cd());
    pipe_refresh();
}

json compute_settings_to_json()
{
    const ComputeDescriptor& cd = get_cd();

    auto j_cs = json{
        {"image rendering",
         {
             {"image mode", computation_to_string[cd.compute_mode.load()]},
             {"batch size", cd.batch_size.load()},
             {"time transformation stride", cd.time_transformation_stride.load()},
             {"filter2d",
              {{"enabled", cd.filter2d_enabled.load()}, {"n1", cd.filter2d_n1.load()}, {"n2", cd.filter2d_n2.load()}}},
             {"space transformation", space_transformation_to_string[cd.space_transformation.load()]},
             {"time transformation", time_transformation_to_string[cd.time_transformation.load()]},
             {"time transformation size", cd.time_transformation_size.load()},
             {"lambda", cd.lambda.load()},
             {"z distance", cd.zdistance.load()},
             {"convolution",
              {{"enabled", cd.convolution_enabled.load()},
               {"type", UserInterfaceDescriptor::instance().convo_name}, // TODO: See user_interface_descriptor.hh (put
                                                                         // var in GSH)
               {"divide", cd.divide_convolution_enabled.load()}}},
         }},
        {"view",
         {
             {"type", img_type_to_string[cd.img_type.load()]},
             {"fft shift", cd.fft_shift_enabled.load()},
             {"x", cd.x.to_json()},
             {"y", cd.y.to_json()},
             {"p", cd.p.to_json()},
             {"q", cd.q.to_json()},
             {"window",
              {{"xy", cd.xy.to_json()},
               {"yz", cd.yz.to_json()},
               {"xz", cd.xz.to_json()},
               {"filter2d", cd.filter2d.to_json()}}},
             {"renorm", cd.renorm_enabled.load()},
             {"reticle", {{"display enabled", cd.reticle_display_enabled.load()}, {"scale", cd.reticle_scale.load()}}},
         }},
        {"composite",
         {
             {"mode", composite_kind_to_string[cd.composite_kind.load()]},
             {"auto weight", cd.composite_auto_weights.load()},
             {"rgb", cd.rgb.to_json()},
             {"hsv", cd.hsv.to_json()},
         }},
        {
            "advanced",
            {{"buffer size",
              {{"input", cd.input_buffer_size.load()},
               {"file", cd.file_buffer_size.load()},
               {"record", cd.record_buffer_size.load()},
               {"output", cd.output_buffer_size.load()},
               {"time transformation cuts", cd.time_transformation_cuts_output_buffer_size.load()}}},
             {
                 "filter2d smooth",
                 {{"low", cd.filter2d_smooth_low.load()}, {"high", cd.filter2d_smooth_high.load()}},
             },
             {"contrast",
              {{"lower", cd.contrast_lower_threshold.load()},
               {"upper", cd.contrast_upper_threshold.load()},
               {"cuts p offset", cd.cuts_contrast_p_offset.load()}}},
             {"renorm constant", cd.renorm_constant.load()}},
        },
    };

    return j_cs;
}

void save_compute_settings(const std::string& json_path)
{
    if (json_path.empty())
        return;

    auto j_cs = compute_settings_to_json();

    std::ofstream file(json_path);
    file << j_cs.dump(1);

    LOG_INFO << "Compute settings overwritten at : " << json_path;
}
} // namespace holovibes::api
