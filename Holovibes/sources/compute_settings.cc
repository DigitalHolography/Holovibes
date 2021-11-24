#include "API.hh"

namespace holovibes::api
{
void load_image_rendering(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    set_compute_mode(static_cast<Computation>(
        ptree.get<int>("image_rendering.image_mode", static_cast<int>(cd.compute_mode.load()))));

    set_batch_size(ptree.get<ushort>("image_rendering.batch_size", cd.batch_size));
    set_time_transformation_stride(
        ptree.get<ushort>("image_rendering.time_transformation_stride", cd.time_transformation_stride));

    set_filter2d(ptree.get<bool>("image_rendering.filter2d_enabled", cd.filter2d_enabled));
    set_filter2d_n1(ptree.get<int>("image_rendering.filter2d_n1", cd.filter2d_n1));
    set_filter2d_n2(ptree.get<int>("image_rendering.filter2d_n2", cd.filter2d_n2));

    set_space_transformation(static_cast<SpaceTransformation>(
        ptree.get<int>("image_rendering.space_transformation", static_cast<int>(cd.space_transformation.load()))));
    set_time_transformation(static_cast<TimeTransformation>(
        ptree.get<int>("image_rendering.time_transformation", static_cast<int>(cd.time_transformation.load()))));

    set_time_transformation_size(
        ptree.get<ushort>("image_rendering.time_transformation_size", cd.time_transformation_size));

    set_wavelength(ptree.get<float>("image_rendering.lambda", cd.lambda));
    set_z_distance(ptree.get<float>("image_rendering.z_distance", cd.zdistance));

    ////// TODO: Remove checkboxe ??
    ////// TODO: Think about how to store the type. Some new convolutions type might be added in AppData
    ////// set_convolution_enabled(ptree.get<bool>("image_rendering.convolution_enabled", cd.convolution_enabled));
    ////// cd.convolution_type( ptree.get("image_rendering.convolution_type", cd.convolution_enabled));
    set_divide_convolution(
        ptree.get<bool>("image_rendering.divide_convolution_enabled", cd.divide_convolution_enabled));
}

void load_view(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    set_img_type(static_cast<ImgType>(ptree.get<int>("view.view_type", static_cast<int>(cd.img_type.load()))));
    // Add unwrap_2d
    set_fft_shift(ptree.get<bool>("view.fft_shift_enabled", cd.fft_shift_enabled));

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

    toggle_renormalize(ptree.get<bool>("view.renorm_enabled", cd.renorm_enabled));

    display_reticle(ptree.get<bool>("view.reticle_display_enabled", cd.reticle_display_enabled));
    reticle_scale(ptree.get<float>("view.reticle_scale", cd.reticle_scale));
}

void load_composite(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    set_composite_kind(
        static_cast<CompositeKind>(ptree.get<int>("composite.mode", static_cast<int>(cd.composite_kind.load()))));
    set_composite_auto_weights(ptree.get<bool>("composite.auto_weights_enabled", cd.composite_auto_weights));

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

void load_view_visibility(const boost::property_tree::ptree& ptree, ComputeDescriptor& cd)
{
    // May just not be in compute_settings.ini because those are settings the user will be able to see
    // if it checked the relevent checbox in the UI.
    // Sets directly using cd because no need of the intern checks of the api function

    get_cd().set_filter2d_view_enabled(
        ptree.get<bool>("image_rendering.filter2d_view_enabled", cd.filter2d_view_enabled));
    get_cd().set_3d_cuts_view_enabled(ptree.get<bool>("view.3d_cuts_enabled", cd.time_transformation_cuts_enabled));
    get_cd().set_lens_view_enabled(ptree.get<bool>("view.lens_view_enabled", cd.lens_view_enabled));
    get_cd().set_raw_view_enabled(ptree.get<bool>("view.raw_view_enabled", cd.raw_view_enabled));
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
    // FIXME: When GSH,            convolution_data["type"]
    cd.divide_convolution_enabled = convolution_data["divide"];
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

    const json& reticle_data = data["window"];
    cd.reticle_display_enabled = reticle_data["display enabled"];
    cd.reticle_scale = reticle_data["scale"];
}

void load_composite(const json& data) { ComputeDescriptor& cd = api::get_cd(); }
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

void load_compute_settings(const std::string& json_path)
{
    if (json_path.empty())
        return;

    LOG_INFO << "Compute settings loaded from : " << json_path;

    auto j_cs = json::parse(json_path);
    json_to_compute_settings(j_cs);

    // boost::property_tree::ptree ptree;
    // boost::property_tree::ini_parser::read_ini(json_path, ptree);
    // load_image_rendering(ptree, get_cd());
    // load_view(ptree, get_cd());
    // load_composite(ptree, get_cd());
    // load_advanced(ptree, get_cd());

    // Currently not working.
    // The app crash when one of the visibility is already set at when the app begins.
    // Possible problem: Concurrency between maindisplay and the other displays
    // load_view_visibility(ptree, get_cd());

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
               {"type", "45"}, // TODO: When GSH will be merged, need a parameter storing name of the file
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
                 "filer2d smooth",
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
