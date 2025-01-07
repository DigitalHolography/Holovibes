#include "transform_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Batch

bool TransformApi::set_batch_size(uint value) const
{
    bool request_time_stride_update = false;
    UPDATE_SETTING(BatchSize, value);

    if (value > api_->input.get_input_buffer_size())
        value = api_->input.get_input_buffer_size();

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

void TransformApi::update_batch_size(uint batch_size) const
{
    if (api_->input.get_data_type() == RecordedDataType::MOMENTS)
        batch_size = 1;

    if (api_->compute.get_is_computation_stopped() || get_batch_size() == batch_size)
        return;

    if (set_batch_size(batch_size))
        api_->compute.get_compute_pipe()->request(ICS::UpdateTimeStride);
    api_->compute.get_compute_pipe()->request(ICS::UpdateBatchSize);
}

#pragma endregion

#pragma region Time Stride

void TransformApi::set_time_stride(uint value) const
{
    UPDATE_SETTING(TimeStride, value);

    uint batch_size = GET_SETTING(BatchSize);

    if (batch_size > value)
        UPDATE_SETTING(TimeStride, batch_size);
    // Go to lower multiple
    if (value % batch_size != 0)
        UPDATE_SETTING(TimeStride, value - value % batch_size);
}

void TransformApi::update_time_stride(const uint time_stride) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || api_->compute.get_is_computation_stopped())
        return;

    if (time_stride == get_time_stride())
        return;

    set_time_stride(time_stride);
    api_->compute.get_compute_pipe()->request(ICS::UpdateTimeStride);
}

#pragma endregion

#pragma region Space Tr.

void TransformApi::set_space_transformation(const SpaceTransformation value) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || get_space_transformation() == value)
        return;

    UPDATE_SETTING(SpaceTransformation, value);
    api_->compute.pipe_refresh();
}

void TransformApi::set_lambda(float value) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    UPDATE_SETTING(Lambda, value < 0 ? 0 : value);
    api_->compute.pipe_refresh();
}

void TransformApi::set_z_distance(float value) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    // Avoid 0 for cuda kernel
    if (value == 0)
        value = 0.000001f;

    UPDATE_SETTING(ZDistance, value);
    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Time Tr.

void TransformApi::update_time_transformation_size(uint time_transformation_size) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || api_->compute.get_is_computation_stopped())
        return;

    if (time_transformation_size == get_time_transformation_size())
        return;

    if (time_transformation_size < 1)
        time_transformation_size = 1;

    set_time_transformation_size(time_transformation_size);
    api_->compute.get_compute_pipe()->request(ICS::UpdateTimeTransformationSize);
}

void TransformApi::set_time_transformation(const TimeTransformation value) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || get_time_transformation() == value)
        return;

    UPDATE_SETTING(TimeTransformation, value);
    api_->composite.set_z_fft_shift(value == TimeTransformation::STFT);
    api_->compute.get_compute_pipe()->request(ICS::UpdateTimeTransformationAlgorithm);
}

#pragma endregion

#pragma region Time Tr.Freq.

void TransformApi::set_p_index(uint value) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    if (value >= get_time_transformation_size() || value == 0)
    {
        LOG_ERROR("p param has to be between 1 and #img");
        return;
    }

    SET_SETTING(P, start, value);
    api_->compute.pipe_refresh();
}

void TransformApi::set_p_accu_level(uint p_value) const
{
    SET_SETTING(P, width, p_value);
    api_->compute.pipe_refresh();
}

void TransformApi::set_q_index(uint value) const
{
    SET_SETTING(Q, start, value);
    api_->compute.pipe_refresh();
}

void TransformApi::set_q_accu_level(uint value) const
{
    SET_SETTING(Q, width, value);
    api_->compute.pipe_refresh();
}

void TransformApi::check_p_limits() const
{
    int upper_bound = static_cast<int>(get_time_transformation_size()) - 1;

    if (std::cmp_greater(get_p_accu_level(), upper_bound))
        set_p_accu_level(upper_bound);

    upper_bound -= get_p_accu_level();

    if (upper_bound >= 0 && get_p_index() > static_cast<uint>(upper_bound))
        set_p_index(upper_bound);
}

void TransformApi::check_q_limits() const
{
    int upper_bound = static_cast<int>(get_time_transformation_size()) - 1;

    if (std::cmp_greater(get_q_accu_level(), upper_bound))
        set_q_accu_level(upper_bound);

    upper_bound -= get_q_accu_level();

    if (upper_bound >= 0 && get_q_index() > static_cast<uint>(upper_bound))
        set_q_index(upper_bound);
}

#pragma endregion

#pragma region Time Tr.Cuts

void TransformApi::set_x_accu_level(uint x_value) const
{
    SET_SETTING(X, width, x_value);
    api_->compute.pipe_refresh();
}

void TransformApi::set_x_cuts(uint value) const
{
    if (value < api_->input.get_input_fd().width)
    {
        SET_SETTING(X, start, value);
        api_->compute.pipe_refresh();
    }
}

void TransformApi::set_y_accu_level(uint y_value) const
{
    SET_SETTING(Y, width, y_value);
    api_->compute.pipe_refresh();
}

void TransformApi::set_y_cuts(uint value) const
{
    if (value < api_->input.get_input_fd().height)
    {
        SET_SETTING(Y, start, value);
        api_->compute.pipe_refresh();
    }
}

void TransformApi::set_x_y(uint x, uint y) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || api_->compute.get_is_computation_stopped())
        return;

    if (x < api_->input.get_input_fd().width)
        SET_SETTING(X, start, x);

    if (y < api_->input.get_input_fd().height)
        SET_SETTING(Y, start, y);

    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Specials

void TransformApi::set_unwrapping_2d(const bool value) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    api_->compute.get_compute_pipe()->set_requested(ICS::Unwrap2D, value);
    api_->compute.pipe_refresh();
}

void TransformApi::set_fft_shift_enabled(bool value) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return;

    UPDATE_SETTING(FftShiftEnabled, value);
    if (api_->global_pp.get_registration_enabled())
        api_->compute.get_compute_pipe()->request(ICS::UpdateRegistrationZone);

    api_->compute.pipe_refresh();
}

#pragma endregion

} // namespace holovibes::api