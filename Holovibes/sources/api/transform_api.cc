#include "transform_api.hh"

namespace holovibes::api
{

#pragma region Batch

bool set_batch_size(uint value)
{
    bool request_time_stride_update = false;
    UPDATE_SETTING(BatchSize, value);

    if (value > get_input_buffer_size())
        value = get_input_buffer_size();

    uint time_stride = get_time_stride();
    if (time_stride < value)
    {
        UPDATE_SETTING(TimeStride, value);
        time_stride = value;
        request_time_stride_update = true;
    }

    // Go to lower multiple
    if (time_stride % value != 0)
    {
        request_time_stride_update = true;
        set_time_stride(time_stride - time_stride % value);
    }

    return request_time_stride_update;
}

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

#pragma region Time Stride

void set_time_stride(uint value)
{
    UPDATE_SETTING(TimeStride, value);

    uint batch_size = GET_SETTING(BatchSize);

    if (batch_size > value)
        UPDATE_SETTING(TimeStride, batch_size);
    // Go to lower multiple
    if (value % batch_size != 0)
        UPDATE_SETTING(TimeStride, value - value % batch_size);
}

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

#pragma region Space Tr.

void set_space_transformation(const SpaceTransformation value)
{
    if (api::get_compute_mode() == Computation::Raw || api::get_space_transformation() == value)
        return;

    UPDATE_SETTING(SpaceTransformation, value);
    pipe_refresh();
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

#pragma endregion

#pragma region Time Tr.

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

void set_time_transformation(const TimeTransformation value)
{
    if (api::get_compute_mode() == Computation::Raw || api::get_time_transformation() == value)
        return;

    UPDATE_SETTING(TimeTransformation, value);
    set_z_fft_shift(value == TimeTransformation::STFT);
    get_compute_pipe()->request(ICS::UpdateTimeTransformationAlgorithm);
}

#pragma endregion

#pragma region Time Tr.Freq.

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

void check_p_limits()
{
    int upper_bound = static_cast<int>(get_time_transformation_size()) - 1;

    if (std::cmp_greater(get_p_accu_level(), upper_bound))
        api::set_p_accu_level(upper_bound);

    upper_bound -= get_p_accu_level();

    if (upper_bound >= 0 && get_p_index() > static_cast<uint>(upper_bound))
        api::set_p_index(upper_bound);
}

void check_q_limits()
{
    int upper_bound = static_cast<int>(get_time_transformation_size()) - 1;

    if (std::cmp_greater(get_q_accu_level(), upper_bound))
        api::set_q_accu_level(upper_bound);

    upper_bound -= get_q_accu_level();

    if (upper_bound >= 0 && get_q_index() > static_cast<uint>(upper_bound))
        api::set_q_index(upper_bound);
}

#pragma endregion

#pragma region Time Tr.Cuts

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

#pragma endregion

#pragma region Specials

void set_unwrapping_2d(const bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    get_compute_pipe()->set_requested(ICS::Unwrap2D, value);
    pipe_refresh();
}

void set_fft_shift_enabled(bool value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    UPDATE_SETTING(FftShiftEnabled, value);
    if (get_registration_enabled())
        api::get_compute_pipe()->request(ICS::UpdateRegistrationZone);
    pipe_refresh();
}

#pragma endregion

} // namespace holovibes::api