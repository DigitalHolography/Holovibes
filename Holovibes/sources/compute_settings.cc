#include "API.hh"

namespace holovibes::api
{
void load_image_rendering(const json& data)
{

    GSH::instance().set_compute_mode(computation_from_string(data["image mode"]));
    GSH::instance().set_batch_size(data["batch size"]);
    GSH::instance().set_time_transformation_stride(data["time transformation stride"]);

    const json& filter_2d_data = data["filter2d"];
    GSH::instance().set_filter2d_enabled(filter_2d_data["enabled"]);
    GSH::instance().set_filter2d_n1(filter_2d_data["n1"]);
    GSH::instance().set_filter2d_n2(filter_2d_data["n2"]);

    GSH::instance().set_space_transformation(space_transformation_from_string(data["space transformation"]));
    GSH::instance().set_time_transformation(time_transformation_from_string(data["time transformation"]));
    GSH::instance().set_time_transformation_size(data["time transformation size"]);
    GSH::instance().set_lambda(data["lambda"]);
    GSH::instance().set_z_distance(data["z distance"]);

    const json& convolution_data = data["convolution"];
    GSH::instance().set_convolution_enabled(convolution_data["enabled"]);
    // FIXME: Use GSH instead of UID
    UserInterfaceDescriptor::instance().convo_name = convolution_data["type"];
    // FIXME: Loads convolution matrix
    GSH::instance().set_divide_convolution_enabled(convolution_data["divide"]);

    // FIXME: Need to use setters that are currently in CD
    // Example, matrix is not loaded if we do not pass through setter
}

void load_view(const json& data)
{
    GSH::instance().set_img_type(img_type_from_string(data["type"]));
    GSH::instance().set_fft_shift_enabled(data["fft shift"]);
    GSH::instance().set_x(View_XY(data["x"]));
    GSH::instance().set_y(View_XY(data["y"]));
    GSH::instance().set_p(View_PQ(data["p"]));
    GSH::instance().set_q(View_PQ(data["q"]));

    const json& window_data = data["window"];
    GSH::instance().set_xy(View_XYZ(window_data["xy"]));
    GSH::instance().set_yz(View_XYZ(window_data["yz"]));
    GSH::instance().set_xz(View_XYZ(window_data["xz"]));
    GSH::instance().set_filter2d(View_Window(window_data["filter2d"]));

    GSH::instance().set_renorm_enabled(data["renorm"]);

    const json& reticle_data = data["reticle"];
    GSH::instance().set_reticle_display_enabled(reticle_data["display enabled"]);
    GSH::instance().set_reticle_scale(reticle_data["scale"]);
}

void load_composite(const json& data)
{
    GSH::instance().set_composite_kind(composite_kind_from_string(data["mode"]));
    GSH::instance().set_composite_auto_weights(data["auto weight"]);
    GSH::instance().set_rgb(Composite_RGB(data["rgb"]));
    GSH::instance().set_hsv(Composite_HSV(data["hsv"]));
}

void load_advanced(const json& data)
{

    const json& buffer_size_data = data["buffer size"];
    GSH::instance().set_file_buffer_size(buffer_size_data["file"]);
    GSH::instance().set_input_buffer_size(buffer_size_data["input"]);
    GSH::instance().set_output_buffer_size(buffer_size_data["output"]);
    GSH::instance().set_record_buffer_size(buffer_size_data["record"]);
    GSH::instance().set_time_transformation_cuts_output_buffer_size(buffer_size_data["time transformation cuts"]);

    const json& contrast_data = data["contrast"];
    GSH::instance().set_contrast_lower_threshold(contrast_data["lower"]);
    GSH::instance().set_contrast_upper_threshold(contrast_data["upper"]);
    GSH::instance().set_cuts_contrast_p_offset(contrast_data["cuts p offset"]);

    const json& filter2d_smooth_data = data["filter2d smooth"];
    GSH::instance().set_filter2d_smooth_high(filter2d_smooth_data["high"]);
    GSH::instance().set_filter2d_smooth_low(filter2d_smooth_data["low"]);

    GSH::instance().set_renorm_constant(data["renorm constant"]);
}

void json_to_compute_settings(const json& data)
{
    load_image_rendering(data["image rendering"]);
    load_view(data["view"]);
    load_composite(data["composite"]);
    load_advanced(data["advanced"]);
}

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
    if (json_path.empty())
        return;

    LOG_INFO << "Compute settings loaded from : " << json_path;

    std::ifstream ifs(json_path);
    auto j_cs = json::parse(ifs);

    json_to_compute_settings(j_cs);

    after_load_checks();
    pipe_refresh();
}

// clang-format off

json compute_settings_to_json()
{

    auto j_cs = json{
        {"image rendering", {
                {"image mode", computation_to_string(GSH::instance().get_compute_mode())},
                {"batch size", GSH::instance().get_batch_size()},
                {"time transformation stride", GSH::instance().get_time_transformation_stride()},
                {"filter2d", {
                        {"enabled", GSH::instance().get_filter2d_enabled()},
                        {"n1", GSH::instance().get_filter2d_n1()},
                        {"n2", GSH::instance().get_filter2d_n2()}
                    }
                },
                {"space transformation", space_transformation_to_string(GSH::instance().get_space_transformation())},
                {"time transformation", time_transformation_to_string(GSH::instance().get_time_transformation())},
                {"time transformation size", GSH::instance().get_time_transformation_size()},
                {"lambda", GSH::instance().get_lambda()},
                {"z distance", GSH::instance().get_z_distance()},
                {"convolution", {
                        {"enabled", GSH::instance().get_convolution_enabled()},
                        {"type", UserInterfaceDescriptor::instance().convo_name},
                        {"divide", GSH::instance().get_divide_convolution_enabled()}
                    }
                },
            }
        },
        {"view", {
                {"type", img_type_to_string(GSH::instance().get_img_type())},
                {"fft shift", GSH::instance().get_fft_shift_enabled()},
                {"x", GSH::instance().get_x()},
                {"y", GSH::instance().get_y()},
                {"p", GSH::instance().get_p()},
                {"q", GSH::instance().get_q()},
                {"window", {
                        {"xy", GSH::instance().get_xy()},
                        {"yz", GSH::instance().get_yz()},
                        {"xz", GSH::instance().get_xz()},
                        {"filter2d", GSH::instance().get_filter2d()}
                    }
                },
                {"renorm", GSH::instance().get_renorm_enabled()},
                {"reticle", {
                        {"display enabled", GSH::instance().get_reticle_display_enabled()},
                        {"scale", GSH::instance().get_reticle_scale()}
                    }
                },
            }
        },
        {"composite", {
                {"mode", composite_kind_to_string(GSH::instance().get_composite_kind())},
                {"auto weight", GSH::instance().get_composite_auto_weights()},
                {"rgb", GSH::instance().get_rgb()},
                {"hsv", GSH::instance().get_hsv()},
            }
        },
        {"advanced", {
                {"buffer size", {
                        {"input", GSH::instance().get_input_buffer_size()},
                        {"file", GSH::instance().get_file_buffer_size()},
                        {"record", GSH::instance().get_record_buffer_size()},
                        {"output", GSH::instance().get_output_buffer_size()},
                        {"time transformation cuts", GSH::instance().get_time_transformation_cuts_output_buffer_size()}
                    }
                },
                {"filter2d smooth", {
                        {"low", GSH::instance().get_filter2d_smooth_low()},
                        {"high", GSH::instance().get_filter2d_smooth_high()}
                    },
                },
                {"contrast", {
                        {"lower", GSH::instance().get_contrast_lower_threshold()},
                        {"upper", GSH::instance().get_contrast_upper_threshold()},
                        {"cuts p offset", GSH::instance().get_cuts_contrast_p_offset()}
                    }
                },
                {"renorm constant", GSH::instance().get_renorm_constant()}
            },
        },
    };

    return j_cs;
}

// clang-format on

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
