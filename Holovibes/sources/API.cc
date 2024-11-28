#include "API.hh"
#include "logger.hh"
#include "notifier.hh"
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

void set_computation_mode(Computation mode)
{
    close_critical_compute();

    set_compute_mode(mode);
    create_pipe();

    if (mode == Computation::Hologram)
    {
        api::change_window(static_cast<int>(WindowKind::XYview));
        api::set_contrast_enabled(true);
    }
    else
        set_record_mode_enum(RecordMode::RAW); // Force set record mode to raw because it cannot be anything else
}

ApiCode set_view_mode(const ImgType type)
{
    if (type == api::get_img_type())
        return ApiCode::NO_CHANGE;

    if (api::get_import_type() == ImportType::None)
        return ApiCode::NOT_STARTED;

    if (api::get_compute_mode() == Computation::Raw)
        return ApiCode::WRONG_MODE;

    try
    {
        bool composite = type == ImgType::Composite || api::get_img_type() == ImgType::Composite;

        api::set_img_type(type);

        // Switching to composite or back from composite needs a recreation of the pipe since buffers size will be *3
        if (composite)
            set_computation_mode(Computation::Hologram);
        else
            pipe_refresh();
    }
    catch (const std::runtime_error&) // The pipe is not initialized
    {
        return ApiCode::FAILURE;
    }

    return ApiCode::OK;
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

#pragma endregion

#pragma region Computation

void handle_update_exception()
{
    api::set_p_index(0);
    api::set_time_transformation_size(1);
    api::disable_convolution();
    api::enable_filter("");
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
}

void set_p_accu_level(uint p_value)
{
    SET_SETTING(P, width, p_value);
    pipe_refresh();
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

#pragma region Contrast - Log

void set_raw_bitshift(unsigned int value) { UPDATE_SETTING(RawBitshift, value); }

unsigned int get_raw_bitshift() { return static_cast<unsigned int>(GET_SETTING(RawBitshift)); }

#pragma endregion

#pragma region Information

void start_information_display() { Holovibes::instance().start_information_display(); }

#pragma endregion

} // namespace holovibes::api
