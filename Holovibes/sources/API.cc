#include "API.hh"
#include "logger.hh"
#include "input_filter.hh"
#include "notifier.hh"
#include "logger.hh"
#include "tools.hh"

#include <string>
#include <unordered_map>

namespace holovibes::api
{

#pragma region Local

void disable_pipe_refresh()
{
    try
    {
        get_compute_pipe()->clear_request(ICS::RefreshEnabled);
    }
    catch (const std::runtime_error&)
    {
    }
}

void enable_pipe_refresh()
{
    try
    {
        get_compute_pipe()->set_requested(ICS::RefreshEnabled, true);
    }
    catch (const std::runtime_error&)
    {
    }
}

void pipe_refresh()
{
    if (get_import_type() == ImportType::None)
        return;

    try
    {
        LOG_TRACE("pipe_refresh");
        get_compute_pipe()->request_refresh();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR("{}", e.what());
    }
}
const QUrl get_documentation_url() { return QUrl("https://ftp.espci.fr/incoming/Atlan/holovibes/manual/"); }

static bool is_current_window_xyz_type()
{
    static const std::set<WindowKind> types = {WindowKind::XYview, WindowKind::XZview, WindowKind::YZview};
    return types.contains(api::get_current_window_type());
}

void close_windows()
{
    if (UserInterfaceDescriptor::instance().mainDisplay.get() != nullptr)
        UserInterfaceDescriptor::instance().mainDisplay.get()->save_gui("holo window");
    UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);

    UserInterfaceDescriptor::instance().sliceXZ.reset(nullptr);
    UserInterfaceDescriptor::instance().sliceYZ.reset(nullptr);
    UserInterfaceDescriptor::instance().filter2d_window.reset(nullptr);
    UserInterfaceDescriptor::instance().lens_window.reset(nullptr);
    UserInterfaceDescriptor::instance().plot_window_.reset(nullptr);
    UserInterfaceDescriptor::instance().raw_window.reset(nullptr);
}

#pragma endregion

#pragma region Close Compute

void camera_none()
{
    close_windows();
    close_critical_compute();

    Holovibes::instance().stop_frame_read();

    set_camera_kind(CameraKind::NONE);
    set_is_computation_stopped(true);
    set_import_type(ImportType::None);
}

#pragma endregion

#pragma region Cameras

bool change_camera(CameraKind c)
{
    LOG_FUNC(static_cast<int>(c));
    camera_none();

    auto path = holovibes::settings::user_settings_filepath;
    LOG_INFO("path: {}", path);
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    j_us["camera"]["type"] = c;

    if (c == CameraKind::NONE)
    {
        std::ofstream output_file(path);
        output_file << j_us.dump(1);
        return false;
    }
    try
    {
        if (get_compute_mode() == Computation::Raw)
            Holovibes::instance().stop_compute();

        set_data_type(RecordedDataType::RAW); // The data gotten from a camera is raw

        try
        {
            Holovibes::instance().start_camera_frame_read(c);
        }
        catch (const std::exception&)
        {
            LOG_INFO("Set camera to NONE");

            j_us["camera"]["type"] = 0;
            std::ofstream output_file(path);
            output_file << j_us.dump(1);
            Holovibes::instance().stop_frame_read();
            return false;
        }

        set_camera_kind(c);
        set_import_type(ImportType::Camera);
        set_is_computation_stopped(false);

        std::ofstream output_file(path);
        output_file << j_us.dump(1);

        return true;
    }
    catch (const camera::CameraException& e)
    {
        LOG_ERROR("[CAMERA] {}", e.what());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }

    std::ofstream output_file(path);
    output_file << j_us.dump(1);
    return false;
}

void configure_camera()
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(Holovibes::instance().get_camera_ini_name())));
}

#pragma endregion

#pragma region Image Mode

void create_pipe()
{
    LOG_FUNC();
    try
    {
        Holovibes::instance().start_compute();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR("cannot create Pipe: {}", e.what());
    }
}

QPoint getSavedHoloWindowPos()
{
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    int x = json_get_or_default(j_us, 0, "holo window", "x");
    int y = json_get_or_default(j_us, 0, "holo window", "y");
    return QPoint(x, y);
}

QSize getSavedHoloWindowSize(ushort& width, ushort& height)
{
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    int final_width = json_get_or_default(j_us, width, "holo window", "width");
    int final_height = json_get_or_default(j_us, height, "holo window", "height");
    return QSize(final_width, final_height);
}

void set_light_ui_mode(bool value)
{
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);
    j_us["light_ui"] = value;

    std::ofstream output_file(path);
    output_file << j_us.dump(1);
}

bool is_light_ui_mode()
{
    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    json j_us = json::parse(input_file);

    return json_get_or_default(j_us, false, "light_ui");
}

void set_computation_mode(Computation mode, uint window_max_size)
{
    close_windows();
    close_critical_compute();

    set_compute_mode(mode);
    create_pipe();

    if (mode == Computation::Hologram)
    {
        api::change_window(static_cast<int>(WindowKind::XYview));
        api::set_contrast_mode(true);
    }

    create_window(mode, window_max_size);
}

void create_window(Computation window_kind, ushort window_size)
{
    const camera::FrameDescriptor& fd = get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, window_size);

    QPoint pos = getSavedHoloWindowPos();
    QSize size = getSavedHoloWindowSize(width, height);

    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition();
        size = UserInterfaceDescriptor::instance().mainDisplay->size();
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
    }

    if (window_kind == Computation::Raw)
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(
            new holovibes::gui::RawWindow(pos,
                                          size,
                                          get_input_queue().get(),
                                          static_cast<float>(width) / static_cast<float>(height)));
        UserInterfaceDescriptor::instance().mainDisplay->setBitshift(get_raw_bitshift());
    }
    else
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(
            new gui::HoloWindow(pos,
                                size,
                                get_gpu_output_queue().get(),
                                get_compute_pipe(),
                                UserInterfaceDescriptor::instance().sliceXZ,
                                UserInterfaceDescriptor::instance().sliceYZ,
                                static_cast<float>(width) / static_cast<float>(height)));
        UserInterfaceDescriptor::instance().mainDisplay->set_is_resize(false);
        UserInterfaceDescriptor::instance().mainDisplay->resetTransform();
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(api::get_rotation());
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(api::get_horizontal_flip());
    }

    UserInterfaceDescriptor::instance().mainDisplay->setTitle(QString("XY view"));
}

void refresh_view_mode(ushort window_size, ImgType img_type)
{
    float old_scale = 1.f;
    glm::vec2 old_translation(0.f, 0.f);
    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        old_scale = UserInterfaceDescriptor::instance().mainDisplay->getScale();
        old_translation = UserInterfaceDescriptor::instance().mainDisplay->getTranslate();
    }

    set_img_type(img_type);
    set_computation_mode(Computation::Hologram, window_size);

    UserInterfaceDescriptor::instance().mainDisplay->setScale(old_scale);
    UserInterfaceDescriptor::instance().mainDisplay->setTranslate(old_translation[0], old_translation[1]);
}

void set_view_mode(const ImgType type)
{
    try
    {
        auto pipe = get_compute_pipe();

        api::set_img_type(type);

        // Force XYview autocontrast
        pipe->request_autocontrast(WindowKind::XYview);

        // Force cuts views autocontrast if needed
        if (api::get_cuts_view_enabled())
            api::set_auto_contrast_cuts();

        pipe_refresh();
    }
    catch (const std::runtime_error&) // The pipe is not initialized
    {
    }
}

#pragma endregion

#pragma region Batch

void update_batch_size(uint batch_size)
{
    if (get_data_type() == RecordedDataType::MOMENTS)
        batch_size = 1;

    if (get_import_type() == ImportType::None || get_batch_size() == batch_size)
        return;

    if (set_batch_size(batch_size))
        api::get_compute_pipe()->request(ICS::UpdateTimeStride);
    api::get_compute_pipe()->request(ICS::UpdateBatchSize);
}

#pragma endregion

#pragma region STFT

void update_time_stride(const uint time_stride)
{
    if (get_compute_mode() == Computation::Raw || get_import_type() == ImportType::None)
        return;

    if (time_stride == get_time_stride())
        return;

    set_time_stride(time_stride);
    get_compute_pipe()->request(ICS::UpdateTimeStride);
}

bool set_3d_cuts_view(bool enabled)
{
    if (api::get_import_type() == ImportType::None)
        return false;

    if (enabled)
    {
        try
        {
            get_compute_pipe()->request(ICS::TimeTransformationCuts);
            while (get_compute_pipe()->is_requested(ICS::TimeTransformationCuts))
                continue;

            set_yz_enabled(true);
            set_xz_enabled(true);
            set_cuts_view_enabled(true);
            set_auto_contrast_cuts();

            pipe_refresh();

            return true;
        }
        catch (const std::logic_error& e)
        {
            LOG_ERROR("Catch {}", e.what());
        }
    }
    else
    {
        set_yz_enabled(false);
        set_xz_enabled(false);
        set_cuts_view_enabled(false);

        get_compute_pipe()->request(ICS::DeleteTimeTransformationCuts);
        while (get_compute_pipe()->is_requested(ICS::DeleteTimeTransformationCuts))
            continue;

        return true;
    }

    return false;
}

#pragma endregion

#pragma region Computation

void change_window(const int index) { UPDATE_SETTING(CurrentWindow, static_cast<WindowKind>(index)); }

void toggle_renormalize(bool value)
{
    set_renorm_enabled(value);

    if (get_import_type() != ImportType::None)
        get_compute_pipe()->request(ICS::ClearImgAccu);

    pipe_refresh();
}

void handle_update_exception()
{
    api::set_p_index(0);
    api::set_time_transformation_size(1);
    api::disable_convolution();
    api::disable_filter();
}

void set_filter2d(bool checked)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    set_filter2d_enabled(checked);
    set_auto_contrast_all();
}

void set_filter2d_view(bool enabled)
{
    if (get_compute_mode() == Computation::Raw || get_import_type() == ImportType::None)
        return;

    auto pipe = get_compute_pipe();
    if (enabled)
    {
        pipe->request(ICS::Filter2DView);
        while (pipe->is_requested(ICS::Filter2DView))
            continue;

        set_filter2d_log_enabled(true);
        pipe->request_autocontrast(WindowKind::Filter2D);
        pipe_refresh();
    }
    else
    {
        pipe->request(ICS::DisableFilter2DView);
        while (pipe->is_requested(ICS::DisableFilter2DView))
            continue;
    }
}

void update_time_transformation_size(uint time_transformation_size)
{
    if (get_compute_mode() == Computation::Raw || get_import_type() == ImportType::None)
        return;

    if (time_transformation_size == api::get_time_transformation_size())
        return;

    if (time_transformation_size < 1)
        time_transformation_size = 1;

    set_time_transformation_size(time_transformation_size);
    get_compute_pipe()->request(ICS::UpdateTimeTransformationSize);
}

void set_chart_display_enabled(bool value) { UPDATE_SETTING(ChartDisplayEnabled, value); }

void set_filter2d_view_enabled(bool value) { UPDATE_SETTING(Filter2dViewEnabled, value); }

void set_lens_view(bool enabled)
{
    if (api::get_import_type() == ImportType::None || get_compute_mode() == Computation::Raw)
        return;

    set_lens_view_enabled(enabled);

    if (!enabled)
    {
        auto pipe = get_compute_pipe();
        pipe->request(ICS::DisableLensView);
        while (pipe->is_requested(ICS::DisableLensView))
            continue;
    }
}

void set_raw_view(bool enabled)
{
    if (get_import_type() == ImportType::None || get_compute_mode() == Computation::Raw)
        return;

    if (enabled && get_batch_size() > get_output_buffer_size())
    {
        LOG_ERROR("[RAW VIEW] Batch size must be lower than output queue size");
        return;
    }

    auto pipe = get_compute_pipe();
    set_raw_view_enabled(enabled);

    auto request = enabled ? ICS::RawView : ICS::DisableRawView;

    pipe->request(request);
    while (pipe->is_requested(request))
        continue;

    pipe_refresh();
}

void set_x_accu_level(uint x_value)
{
    SET_SETTING(X, width, x_value);
    pipe_refresh();
}

void set_x_cuts(uint value)
{
    if (value < get_fd().width)
    {
        SET_SETTING(X, start, value);
        pipe_refresh();
    }
}

void set_y_accu_level(uint y_value)
{
    SET_SETTING(Y, width, y_value);
    pipe_refresh();
}

void set_y_cuts(uint value)
{
    if (value < get_fd().height)
    {
        SET_SETTING(Y, start, value);
        pipe_refresh();
    }
}

void set_x_y(uint x, uint y)
{
    if (get_compute_mode() == Computation::Raw || get_import_type() == ImportType::None)
        return;

    if (x < get_fd().width)
        SET_SETTING(X, start, x);

    if (y < get_fd().width)
        SET_SETTING(Y, start, y);

    pipe_refresh();
}

void set_q_index(uint value)
{
    SET_SETTING(Q, start, value);
    pipe_refresh();
}

void set_q_accu_level(uint value)
{
    SET_SETTING(Q, width, value);
    pipe_refresh();
}

void set_p_index(uint value)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    if (value >= get_time_transformation_size() || value == 0)
    {
        LOG_ERROR("p param has to be between 1 and #img");
        return;
    }

    SET_SETTING(P, start, value);
    pipe_refresh();
    set_auto_contrast();
}

void set_p_accu_level(uint p_value)
{
    SET_SETTING(P, width, p_value);
    pipe_refresh();
}

void set_composite_intervals(int composite_p_red, int composite_p_blue)
{
    holovibes::CompositeRGB rgb = GET_SETTING(RGB);
    rgb.frame_index.min = composite_p_red;
    rgb.frame_index.max = composite_p_blue;
    UPDATE_SETTING(RGB, rgb);
    pipe_refresh();
}

void set_composite_intervals_hsv_h_min(uint composite_p_min_h)
{
    set_composite_p_h(composite_p_min_h, get_composite_p_max_h());
    pipe_refresh();
}

void set_composite_intervals_hsv_h_max(uint composite_p_max_h)
{
    set_composite_p_h(get_composite_p_min_h(), composite_p_max_h);
    pipe_refresh();
}

void set_composite_intervals_hsv_s_min(uint composite_p_min_s)
{
    set_composite_p_min_s(composite_p_min_s);
    pipe_refresh();
}

void set_composite_intervals_hsv_s_max(uint composite_p_max_s)
{
    set_composite_p_max_s(composite_p_max_s);
    pipe_refresh();
}

void set_composite_intervals_hsv_v_min(uint composite_p_min_v)
{
    set_composite_p_min_v(composite_p_min_v);
    pipe_refresh();
}

void set_composite_intervals_hsv_v_max(uint composite_p_max_v)
{
    set_composite_p_max_v(composite_p_max_v);
    pipe_refresh();
}

void set_composite_weights(double weight_r, double weight_g, double weight_b)
{
    set_weight_rgb(weight_r, weight_g, weight_b);
    pipe_refresh();
}

void select_composite_rgb() { set_composite_kind(CompositeKind::RGB); }

void select_composite_hsv() { set_composite_kind(CompositeKind::HSV); }

void actualize_frequency_channel_s(bool composite_p_activated_s)
{
    set_composite_p_activated_s(composite_p_activated_s);
}

void actualize_frequency_channel_v(bool composite_p_activated_v)
{
    set_composite_p_activated_v(composite_p_activated_v);
}

void check_p_limits()
{
    int upper_bound = get_time_transformation_size() - 1;

    if (get_p_accu_level() > upper_bound)
        api::set_p_accu_level(upper_bound);

    upper_bound -= get_p_accu_level();

    if (upper_bound >= 0 && get_p_index() > static_cast<uint>(upper_bound))
        api::set_p_index(upper_bound);
}

void check_q_limits()
{
    int upper_bound = get_time_transformation_size() - 1;

    if (std::cmp_greater(get_q_accu_level(), upper_bound))
        api::set_q_accu_level(upper_bound);

    upper_bound -= get_q_accu_level();

    if (upper_bound >= 0 && get_q_index() > static_cast<uint>(upper_bound))
        api::set_q_index(upper_bound);
}

bool slide_update_threshold(
    const int slider_value, float& receiver, float& bound_to_update, const float lower_bound, const float upper_bound)
{
    receiver = slider_value / 1000.0f;

    if (lower_bound > upper_bound)
    {
        // FIXME bound_to_update = receiver ?
        bound_to_update = slider_value / 1000.0f;

        return true;
    }

    return false;
}

void set_lambda(float value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    UPDATE_SETTING(Lambda, value < 0 ? 0 : value);
    pipe_refresh();
}

void set_z_distance(float value)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    // Avoid 0 for cuda kernel
    if (value <= 0)
        value = 0.000001f;

    UPDATE_SETTING(ZDistance, value);
    pipe_refresh();
}

void set_space_transformation(const SpaceTransformation value)
{
    if (api::get_compute_mode() == Computation::Raw || api::get_space_transformation() == value)
        return;

    UPDATE_SETTING(SpaceTransformation, value);
    pipe_refresh();
}

void set_time_transformation(const TimeTransformation value)
{
    if (api::get_compute_mode() == Computation::Raw || api::get_time_transformation() == value)
        return;

    UPDATE_SETTING(TimeTransformation, value);
    set_z_fft_shift(value == TimeTransformation::STFT);
    get_compute_pipe()->request(ICS::UpdateTimeTransformationAlgorithm);
}

void set_unwrapping_2d(const bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    get_compute_pipe()->request(ICS::Unwrap2D);
}

WindowKind get_current_window_type() { return GET_SETTING(CurrentWindow); }

ViewWindow get_current_window()
{
    WindowKind window = get_current_window_type();
    if (window == WindowKind::XYview)
        return get_xy();
    else if (window == WindowKind::XZview)
        return get_xz();
    else if (window == WindowKind::YZview)
        return get_yz();
    else
        return get_filter2d();
}

void close_critical_compute()
{
    if (get_convolution_enabled())
        disable_convolution();

    if (get_cuts_view_enabled())
        set_3d_cuts_view(false);

    if (get_filter2d_view_enabled())
        set_filter2d_view(false);

    if (get_lens_view_enabled())
        set_lens_view(false);

    if (get_raw_view_enabled())
        set_raw_view(false);

    Holovibes::instance().stop_compute();
}

void stop_all_worker_controller() { Holovibes::instance().stop_all_worker_controller(); }

int get_input_queue_fd_width() { return get_fd().width; }

int get_input_queue_fd_height() { return get_fd().height; }

float get_boundary() { return Holovibes::instance().get_boundary(); }

#pragma endregion

#pragma region Texture

static void change_angle()
{
    double rot = api::get_rotation();
    double new_rot = (rot == 270.f) ? 0.f : rot + 90.f;

    api::set_rotation(new_rot);
}

void rotateTexture()
{
    change_angle();
    WindowKind window = get_current_window_type();
    if (window == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(get_xy_rotation());
    else if (UserInterfaceDescriptor::instance().sliceXZ && window == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(get_xz_rotation());
    else if (UserInterfaceDescriptor::instance().sliceYZ && window == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(get_yz_rotation());
}

void set_horizontal_flip()
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    set_xyz_member(api::set_xy_horizontal_flip,
                   api::set_xz_horizontal_flip,
                   api::set_yz_horizontal_flip,
                   !api::get_horizontal_flip());
}

void flipTexture()
{
    set_horizontal_flip();
    WindowKind window = get_current_window_type();
    if (window == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(get_xy_horizontal_flip());
    else if (UserInterfaceDescriptor::instance().sliceXZ && window == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(get_xz_horizontal_flip());
    else if (UserInterfaceDescriptor::instance().sliceYZ && window == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(get_yz_horizontal_flip());
}

#pragma endregion

#pragma region Contrast - Log

void set_contrast_mode(bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    auto window = api::get_current_window_type();

    if (window == WindowKind::Filter2D)
        api::set_filter2d_contrast_enabled(value);
    else
        set_xyz_member(api::set_xy_contrast_enabled, api::set_xz_contrast_enabled, api::set_yz_contrast_enabled, value);
    pipe_refresh();
}

void set_auto_contrast_cuts()
{
    auto pipe = get_compute_pipe();
    pipe->request_autocontrast(WindowKind::XZview);
    pipe->request_autocontrast(WindowKind::YZview);
}

bool set_auto_contrast()
{
    if (api::get_compute_mode() == Computation::Raw || !api::get_contrast_enabled())
        return false;

    try
    {
        get_compute_pipe()->request_autocontrast(get_current_window_type());
        return true;
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }

    return false;
}

void set_auto_contrast_all()
{
    if (get_import_type() == ImportType::None)
        return;

    auto pipe = get_compute_pipe();
    pipe->request_autocontrast(WindowKind::XYview);
    if (api::get_cuts_view_enabled())
    {
        pipe->request_autocontrast(WindowKind::XZview);
        pipe->request_autocontrast(WindowKind::YZview);
    }
    if (get_filter2d_view_enabled())
        pipe->request_autocontrast(WindowKind::Filter2D);

    pipe_refresh();
}

void set_contrast_min(float value)
{
    if (api::get_compute_mode() == Computation::Raw || !api::get_contrast_enabled())
        return;

    // Get the minimum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_min();

    if (old_val != value)
    {
        auto window = api::get_current_window_type();
        float new_val = api::get_current_window().log_enabled ? value : pow(10, value);
        if (window == WindowKind::Filter2D)
            api::set_filter2d_contrast_min(new_val);
        else
            set_xyz_member(api::set_xy_contrast_min, api::set_xz_contrast_min, api::set_yz_contrast_min, new_val);
        pipe_refresh();
    }
}

void set_contrast_max(float value)
{
    if (api::get_compute_mode() == Computation::Raw || !api::get_contrast_enabled())
        return;

    // Get the maximum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_max();

    if (old_val != value)
    {
        auto window = api::get_current_window_type();
        float new_val = api::get_current_window().log_enabled ? value : pow(10, value);
        if (window == WindowKind::Filter2D)
            api::set_filter2d_contrast_max(new_val);
        else
            set_xyz_member(api::set_xy_contrast_max, api::set_xz_contrast_max, api::set_yz_contrast_max, new_val);
        pipe_refresh();
    }
}

void set_contrast_invert(bool value)
{
    if (api::get_compute_mode() == Computation::Raw || !api::get_contrast_enabled())
        return;

    auto window = api::get_current_window_type();
    if (window == WindowKind::Filter2D)
        api::set_filter2d_contrast_invert(value);
    else
        set_xyz_member(api::set_xy_contrast_invert, api::set_xz_contrast_invert, api::set_yz_contrast_invert, value);
    pipe_refresh();
}

void set_contrast_auto_refresh(bool value)
{
    if (api::get_compute_mode() == Computation::Raw || !api::get_contrast_enabled())
        return;

    auto window = api::get_current_window_type();
    if (window == WindowKind::Filter2D)
        api::set_filter2d_contrast_auto_refresh(value);
    else
        set_xyz_member(api::set_xy_contrast_auto_refresh,
                       api::set_xz_contrast_auto_refresh,
                       api::set_yz_contrast_auto_refresh,
                       value);
    pipe_refresh();
}

void update_contrast(WindowKind kind, float min, float max)
{
    switch (kind)
    {
    case WindowKind::XYview:
        api::set_xy_contrast(min, max);
        break;
    case WindowKind::XZview:
        api::set_xz_contrast(min, max);
        break;
    case WindowKind::YZview:
        api::set_yz_contrast(min, max);
        break;
    // TODO : set_filter2d_contrast_auto
    default:
        api::set_filter2d_contrast(min, max);
        break;
    }
}

void set_log_scale(const bool value)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    auto window = api::get_current_window_type();
    if (window == WindowKind::Filter2D)
        api::set_filter2d_log_enabled(value);
    else
        set_xyz_member(api::set_xy_log_enabled, api::set_xz_log_enabled, api::set_yz_log_enabled, value);
    if (value && api::get_contrast_enabled())
        set_auto_contrast();

    pipe_refresh();
}

void set_raw_bitshift(unsigned int value) { UPDATE_SETTING(RawBitshift, value); }

unsigned int get_raw_bitshift() { return static_cast<unsigned int>(GET_SETTING(RawBitshift)); }

float get_contrast_min()
{
    bool log_enabled =
        get_view_member(get_filter2d_log_enabled(), get_xy_log_enabled(), get_xz_log_enabled(), get_yz_log_enabled());
    float contrast_min = get_view_member(get_filter2d_contrast_min(),
                                         get_xy_contrast_min(),
                                         get_xz_contrast_min(),
                                         get_yz_contrast_min());
    return log_enabled ? contrast_min : log10(contrast_min);
}

float get_contrast_max()
{
    bool log_enabled =
        get_view_member(get_filter2d_log_enabled(), get_xy_log_enabled(), get_xz_log_enabled(), get_yz_log_enabled());
    float contrast_max = get_view_member(get_filter2d_contrast_max(),
                                         get_xy_contrast_max(),
                                         get_xz_contrast_max(),
                                         get_yz_contrast_max());

    return log_enabled ? contrast_max : log10(contrast_max);
}

bool get_contrast_invert()
{
    return get_view_member(get_filter2d_contrast_invert(),
                           get_xy_contrast_invert(),
                           get_xz_contrast_invert(),
                           get_yz_contrast_invert());
}

bool get_contrast_auto_refresh()
{
    return get_view_member(get_filter2d_contrast_auto_refresh(),
                           get_xy_contrast_auto_refresh(),
                           get_xz_contrast_auto_refresh(),
                           get_yz_contrast_auto_refresh());
}

bool get_contrast_enabled()
{
    return get_view_member(get_filter2d_contrast_enabled(),
                           get_xy_contrast_enabled(),
                           get_xz_contrast_enabled(),
                           get_yz_contrast_enabled());
}

double get_rotation()
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    return get_xyz_member(api::get_xy_rotation(), api::get_xz_rotation(), api::get_yz_rotation());
}

bool get_horizontal_flip()
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    return get_xyz_member(api::get_xy_horizontal_flip(), api::get_xz_horizontal_flip(), api::get_yz_horizontal_flip());
}

bool get_log_enabled()
{
    return get_view_member(get_filter2d_log_enabled(),
                           get_xy_log_enabled(),
                           get_xz_log_enabled(),
                           get_yz_log_enabled());
}

unsigned get_accumulation_level()
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    return get_xyz_member(api::get_xy_accumulation_level(),
                          api::get_xz_accumulation_level(),
                          api::get_yz_accumulation_level());
}

void set_accumulation_level(int value)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");
    set_xyz_members(api::set_xy_accumulation_level,
                    api::set_xz_accumulation_level,
                    api::set_yz_accumulation_level,
                    value);

    pipe_refresh();
}

void set_rotation(double value)
{
    if (!is_current_window_xyz_type())
        throw std::runtime_error("bad window type");

    set_xyz_member(api::set_xy_rotation, api::set_xz_rotation, api::set_yz_rotation, value);

    pipe_refresh();
}

float get_truncate_contrast_max(const int precision)
{
    float value = get_contrast_max();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

float get_truncate_contrast_min(const int precision)
{
    float value = get_contrast_min();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

#pragma endregion

#pragma region Convolution

static inline const std::filesystem::path dir(GET_EXE_DIR);

void load_convolution_matrix_file(const std::string& file, std::vector<float>& convo_matrix)
{
    auto& holo = Holovibes::instance();

    auto path_file = dir / __CONVOLUTION_KERNEL_FOLDER_PATH__ / file; //"convolution_kernels" / file;
    std::string path = path_file.string();

    std::vector<float> matrix;
    uint matrix_width = 0;
    uint matrix_height = 0;
    uint matrix_z = 1;

    // Doing this the C way because it's faster
    FILE* c_file;
    fopen_s(&c_file, path.c_str(), "r");

    if (c_file == nullptr)
    {
        fclose(c_file);
        throw std::runtime_error("Invalid file path");
    }

    // Read kernel dimensions
    if (fscanf_s(c_file, "%u %u %u;", &matrix_width, &matrix_height, &matrix_z) != 3)
    {
        fclose(c_file);
        throw std::runtime_error("Invalid kernel dimensions");
    }

    size_t matrix_size = matrix_width * matrix_height * matrix_z;
    matrix.resize(matrix_size);

    // Read kernel values
    for (size_t i = 0; i < matrix_size; ++i)
    {
        if (fscanf_s(c_file, "%f", &matrix[i]) != 1)
        {
            fclose(c_file);
            throw std::runtime_error("Missing values");
        }
    }

    fclose(c_file);

    // Reshape the vector as a (nx,ny) rectangle, keeping z depth
    const uint output_width = holo.get_gpu_output_queue()->get_fd().width;
    const uint output_height = holo.get_gpu_output_queue()->get_fd().height;
    const uint size = output_width * output_height;

    // The convo matrix is centered and padded with 0 since the kernel is
    // usally smaller than the output Example: kernel size is (2, 2) and
    // output size is (4, 4) The kernel is represented by 'x' and
    //  | 0 | 0 | 0 | 0 |
    //  | 0 | x | x | 0 |
    //  | 0 | x | x | 0 |
    //  | 0 | 0 | 0 | 0 |
    const uint first_col = (output_width / 2) - (matrix_width / 2);
    const uint last_col = (output_width / 2) + (matrix_width / 2);
    const uint first_row = (output_height / 2) - (matrix_height / 2);
    const uint last_row = (output_height / 2) + (matrix_height / 2);

    convo_matrix.resize(size, 0.0f);

    uint kernel_indice = 0;
    for (uint i = first_row; i < last_row; i++)
    {
        for (uint j = first_col; j < last_col; j++)
        {
            (convo_matrix)[i * output_width + j] = matrix[kernel_indice];
            kernel_indice++;
        }
    }
}

void load_convolution_matrix(std::optional<std::string> filename)
{
    api::set_convolution_enabled(true);
    api::set_convo_matrix({});

    // There is no file None.txt for convolution
    if (!filename || filename.value() == UID_CONVOLUTION_TYPE_DEFAULT)
        return;
    std::vector<float> convo_matrix = api::get_convo_matrix();
    const std::string& file = filename.value();

    try
    {
        load_convolution_matrix_file(file, convo_matrix);
        api::set_convo_matrix(convo_matrix);
    }
    catch (std::exception& e)
    {
        api::set_convo_matrix({});
        LOG_ERROR("Couldn't load convolution matrix : {}", e.what());
    }
}

void enable_convolution(const std::string& filename)
{
    if (api::get_import_type() == ImportType::None || !api::get_convolution_enabled())
        return;

    load_convolution_matrix(filename == UID_CONVOLUTION_TYPE_DEFAULT ? std::nullopt : std::make_optional(filename));

    if (filename == UID_CONVOLUTION_TYPE_DEFAULT)
    {
        // Refresh because the current convolution might have change.
        pipe_refresh();
        return;
    }

    try
    {
        auto pipe = get_compute_pipe();
        pipe->request(ICS::Convolution);
        // Wait for the convolution to be enabled for notify
        while (pipe->is_requested(ICS::Convolution))
            continue;
    }
    catch (const std::exception& e)
    {
        disable_convolution();
        LOG_ERROR("Catch {}", e.what());
    }
}

void disable_convolution()
{
    set_convo_matrix({});
    set_convolution_enabled(false);
    try
    {
        auto pipe = get_compute_pipe();
        pipe->request(ICS::DisableConvolution);
        while (pipe->is_requested(ICS::DisableConvolution))
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }
}

void set_divide_convolution(const bool value)
{
    if (get_import_type() == ImportType::None || get_divide_convolution_enabled() == value ||
        !get_convolution_enabled())
        return;

    set_divide_convolution_enabled(value);
    pipe_refresh();
}

#pragma endregion

#pragma region Filter

std::vector<float> get_input_filter() { return GET_SETTING(InputFilter); }

void set_input_filter(std::vector<float> value) { UPDATE_SETTING(InputFilter, value); }

void load_input_filter(std::vector<float> input_filter, const std::string& file)
{
    auto& holo = Holovibes::instance();
    try
    {
        auto path_file = dir / __INPUT_FILTER_FOLDER_PATH__ / file;
        InputFilter(input_filter,
                    path_file.string(),
                    holo.get_gpu_output_queue()->get_fd().width,
                    holo.get_gpu_output_queue()->get_fd().height);
    }
    catch (std::exception& e)
    {
        api::set_input_filter({});
        LOG_ERROR("Couldn't load input filter : {}", e.what());
    }
    pipe_refresh();
}

void enable_filter(const std::string& filename)
{
    UserInterfaceDescriptor::instance().filter_name = filename;
    enable_filter();
}

void enable_filter()
{
    auto filename = UserInterfaceDescriptor::instance().filter_name;
    auto file = filename == UID_FILTER_TYPE_DEFAULT ? std::nullopt : std::make_optional(filename);

    UPDATE_SETTING(FilterEnabled, true);
    set_input_filter({});

    if (get_compute_pipe_no_throw() != nullptr)
    {
        // There is no file None.txt for filtering
        if (file && file.value() != UID_FILTER_TYPE_DEFAULT)
            load_input_filter(get_input_filter(), file.value());
        else
            disable_filter();

        // Refresh because the current filter might have change.
        // pipe_refresh();

        if (filename == UID_FILTER_TYPE_DEFAULT)
            return;

        pipe_refresh();
    }
}

void disable_filter()
{
    set_input_filter({});
    UPDATE_SETTING(FilterEnabled, false);
    try
    {
        auto pipe = get_compute_pipe();
        pipe->request(ICS::DisableFilter);
        while (pipe->is_requested(ICS::DisableFilter))
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }
}

#pragma endregion

#pragma region Reticle

void display_reticle(bool value)
{
    if (get_reticle_display_enabled() == value)
        return;

    set_reticle_display_enabled(value);

    if (value)
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().enable<gui::Reticle>(false);
    else
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable(gui::Reticle);

    pipe_refresh();
}

void reticle_scale(float value)
{
    if (!is_between(value, 0.f, 1.f))
        return;

    set_reticle_scale(value);
    pipe_refresh();
}

void update_registration_zone(float value)
{
    if (!is_between(value, 0.f, 1.f) || api::get_import_type() == ImportType::None)
        return;

    set_registration_zone(value);
    api::get_compute_pipe()->request(ICS::UpdateRegistrationZone);
    pipe_refresh();
}

#pragma endregion

#pragma region Chart

void set_chart_display(bool enabled)
{
    if (get_chart_display_enabled() == enabled)
        return;

    try
    {
        auto pipe = get_compute_pipe();
        auto request = enabled ? ICS::ChartDisplay : ICS::DisableChartDisplay;

        pipe->request(request);
        while (pipe->is_requested(request))
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }
}

#pragma endregion

#pragma region Record

void set_record_buffer_size(uint value)
{
    // since this function is always triggered when we save the advanced settings, even if the location was not modified
    if (get_record_buffer_size() != value)
    {
        UPDATE_SETTING(RecordBufferSize, value);

        if (is_recording())
            stop_record();

        Holovibes::instance().init_record_queue();
    }
}

void set_record_queue_location(Device device)
{
    // we check since this function is always triggered when we save the advanced settings, even if the location was not
    // modified
    if (get_record_queue_location() != device)
    {
        UPDATE_SETTING(RecordQueueLocation, device);

        if (is_recording())
            stop_record();

        Holovibes::instance().init_record_queue();
    }
}

void set_record_mode(const std::string& text)
{
    LOG_FUNC(text);

    // Mapping from string to RecordMode
    static const std::unordered_map<std::string, RecordMode> recordModeMap = {{"Chart", RecordMode::CHART},
                                                                              {"Processed Image", RecordMode::HOLOGRAM},
                                                                              {"Raw Image", RecordMode::RAW},
                                                                              {"3D Cuts XZ", RecordMode::CUTS_XZ},
                                                                              {"3D Cuts YZ", RecordMode::CUTS_YZ},
                                                                              {"Moments", RecordMode::MOMENTS}};

    auto it = recordModeMap.find(text);
    if (it == recordModeMap.end())
    {
        LOG_ERROR("Unknown record mode {}", text);
        throw std::runtime_error("Record mode not handled");
    }

    set_record_mode(it->second);
    RecordMode record_mode = api::get_record_mode();

    if (record_mode != RecordMode::RAW)
        api::set_record_on_gpu(true);

    // Attempt to initialize compute pipe for non-CHART record modes
    if (record_mode != RecordMode::CHART)
    {
        try
        {
            auto pipe = get_compute_pipe();
            if (is_recording())
                stop_record();

            Holovibes::instance().init_record_queue();
            LOG_DEBUG("Pipe initialized");
        }
        catch (const std::exception& e)
        {
            (void)e; // Suppress warning in case debug log is disabled
            LOG_DEBUG("Pipe not initialized: {}", e.what());
        }
    }
}

bool is_recording() { return Holovibes::instance().is_recording(); }

bool start_record_preconditions()
{
    std::optional<size_t> nb_frames_to_record = api::get_record_frame_count();
    bool nb_frame_checked = nb_frames_to_record.has_value();

    if (!nb_frame_checked)
        nb_frames_to_record = std::nullopt;

    if (get_record_mode() == RecordMode::CHART && nb_frames_to_record == std::nullopt)
    {
        LOG_ERROR("Number of frames must be activated");
        return false;
    }

    return true;
}

void set_record_device(const Device device)
{
    if (get_compute_mode() == Computation::Hologram)
        Holovibes::instance().stop_compute();

    if (get_raw_view_queue_location() != device)
        set_raw_view_queue_location(device);

    // We only move the queue from gpu to cpu, since by default the record queue is on the cpu
    if (get_record_queue_location() != device && device == Device::CPU)
        set_record_queue_location(device);

    if (get_input_queue_location() != device)
    {
        ImportType it = get_import_type();

        auto c = CameraKind::NONE;
        if (it == ImportType::Camera)
        {
            c = get_camera_kind();
            camera_none();
        }
        else if (it == ImportType::File)
            import_stop();

        set_input_queue_location(device);

        if (device == Device::CPU)
            set_compute_mode(Computation::Raw);

        if (it == ImportType::Camera)
            change_camera(c);
        else
            import_start();

        set_computation_mode(get_compute_mode(), 1);
    }
}

void start_record(std::function<void()> callback)
{
    if (!start_record_preconditions()) // Check if the record can be started
        return;

    RecordMode record_mode = GET_SETTING(RecordMode);

    if (record_mode == RecordMode::CHART)
        Holovibes::instance().start_chart_record(callback);
    else
        Holovibes::instance().start_frame_record(callback);

    // Notify the changes
    NotifierManager::notify<RecordMode>("record_start", record_mode); // notifying lightUI
    NotifierManager::notify<bool>("acquisition_started", true);       // notifying MainWindow
}

void stop_record()
{
    LOG_FUNC();

    auto record_mode = GET_SETTING(RecordMode);

    if (record_mode == RecordMode::CHART)
        Holovibes::instance().stop_chart_record();
    else if (record_mode != RecordMode::NONE)
        Holovibes::instance().stop_frame_record();

    // Holovibes::instance().get_record_queue().load()->dequeue(-1);

    // Notify the changes
    NotifierManager::notify<RecordMode>("record_stop", record_mode);
}

void record_finished()
{
    // if the record was on the cpu, we have to put the queues on gpu again
    if (api::get_record_on_gpu() == false)
        api::set_record_device(Device::GPU);
}

#pragma endregion

#pragma region Import

void import_stop()
{
    if (api::get_import_type() == ImportType::None)
        return;

    LOG_FUNC();

    close_windows();
    close_critical_compute();

    Holovibes::instance().stop_all_worker_controller();
    Holovibes::instance().start_information_display();

    set_is_computation_stopped(true);
    set_import_type(ImportType::None);
}

bool import_start()
{
    LOG_FUNC();

    // Check if computation is currently running
    if (!api::get_is_computation_stopped())
        import_stop();

    // Because we are in file mode
    camera_none();
    set_is_computation_stopped(false);

    // if the file is to be imported in GPU, we should load the buffer preset for such case
    NotifierManager::notify<bool>(api::get_load_file_in_gpu() ? "set_preset_file_gpu" : "import_start", true);

    try
    {
        Holovibes::instance().init_input_queue(UserInterfaceDescriptor::instance().file_fd_,
                                               api::get_input_buffer_size());
        Holovibes::instance().start_file_frame_read();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
        Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();
        return false;
    }

    set_import_type(ImportType::File);

    return true;
}

std::optional<io_files::InputFrameFile*> import_file(const std::string& filename)
{
    if (!filename.empty())
        // Throw if the file format cannot be handled
        return io_files::InputFrameFileFactory::open(filename);

    return std::nullopt;
}

void set_input_file_start_index(size_t value)
{
    UPDATE_SETTING(InputFileStartIndex, value);
    if (value >= get_input_file_end_index())
        set_input_file_end_index(value + 1);
}

void set_input_file_end_index(size_t value)
{
    UPDATE_SETTING(InputFileEndIndex, value);
    if (value <= get_input_file_start_index())
        set_input_file_start_index(value - 1);
}

#pragma endregion

#pragma region Information

void start_information_display() { Holovibes::instance().start_information_display(); }

#pragma endregion

} // namespace holovibes::api
