#include "transform_api.hh"

#include "API.hh"

#define NOT_SAME_AND_NOT_RAW(old_val, new_val)                                                                         \
    if (old_val == new_val)                                                                                            \
        return ApiCode::NO_CHANGE;                                                                                     \
    if (api_->compute.get_compute_mode() == Computation::Raw)                                                          \
        return ApiCode::WRONG_COMP_MODE;

namespace holovibes::api
{

#pragma region Batch

ApiCode TransformApi::set_batch_size(uint batch_size) const
{
    if (get_batch_size() == batch_size)
        return ApiCode::NO_CHANGE;

    if (api_->input.get_data_type() == RecordedDataType::MOMENTS)
    {
        LOG_WARN("File is in moments mode, batch size is fixed to 3");
        batch_size = 3;
    }

    if (batch_size > api_->input.get_input_buffer_size())
    {
        batch_size = api_->input.get_input_buffer_size();
        LOG_WARN("Batch size cannot be greater than the input queue. Setting it to the input queue size: {}",
                 batch_size);
    }

    UPDATE_SETTING(BatchSize, batch_size);

    // Adjust value of time stride if needed
    set_time_stride(get_time_stride());

    if (api_->compute.get_is_computation_stopped())
        return ApiCode::OK;

    api_->compute.get_compute_pipe()->request(ICS::UpdateBatchSize);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Time Stride

ApiCode TransformApi::set_time_stride(uint time_stride) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw)
        return ApiCode::WRONG_COMP_MODE;

    uint batch_size = GET_SETTING(BatchSize);

    if (batch_size > time_stride)
    {
        LOG_WARN("Time stride cannot be lower than the batch size. Setting it to the batch size: {}", batch_size);
        time_stride = batch_size;
    }

    if (time_stride % batch_size != 0)
    {
        time_stride = time_stride - time_stride % batch_size;
        LOG_WARN("Time stride has to be a multiple of the batch size. Setting it to the lower multiple: {}",
                 time_stride);
    }

    // Check after adjustments that the time stride is different
    if (time_stride == get_time_stride())
        return ApiCode::NO_CHANGE;

    UPDATE_SETTING(TimeStride, time_stride);

    if (api_->compute.get_is_computation_stopped())
        return ApiCode::OK;

    api_->compute.get_compute_pipe()->request(ICS::UpdateTimeStride);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Space Tr.

ApiCode TransformApi::set_space_transformation(const SpaceTransformation value) const
{
    NOT_SAME_AND_NOT_RAW(get_space_transformation(), value);

    UPDATE_SETTING(SpaceTransformation, value);
    api_->compute.pipe_refresh();

    return ApiCode::OK;
}

ApiCode TransformApi::set_lambda(float value) const
{
    NOT_SAME_AND_NOT_RAW(get_lambda(), value);

    if (value < 0)
    {
        LOG_WARN("Lambda cannot be negative. Setting it to 0");
        value = 0;
    }

    UPDATE_SETTING(Lambda, value);
    api_->compute.pipe_refresh();

    return ApiCode::OK;
}

ApiCode TransformApi::set_z_distance(float value) const
{
    NOT_SAME_AND_NOT_RAW(get_z_distance(), value);

    // Avoid 0 for cuda kernel
    if (value == 0)
        value = 0.000001f;

    UPDATE_SETTING(ZDistance, value);
    api_->compute.pipe_refresh();

    return ApiCode::OK;
}

#pragma endregion

#pragma region Time Tr.

ApiCode TransformApi::set_time_transformation_size(uint time_transformation_size) const
{
    NOT_SAME_AND_NOT_RAW(get_time_transformation_size(), time_transformation_size);

    if (time_transformation_size < 1)
    {
        LOG_WARN("Time transformation size has to be greater than 0, set to 1");
        time_transformation_size = 1;
    }

    UPDATE_SETTING(TimeTransformationSize, time_transformation_size);

    // Updates p and q bounds
    check_p_limits();
    check_q_limits();

    if (api_->compute.get_is_computation_stopped())
        return ApiCode::OK;

    api_->compute.get_compute_pipe()->request(ICS::UpdateTimeTransformationSize);

    return ApiCode::OK;
}

ApiCode TransformApi::set_time_transformation(const TimeTransformation value) const
{
    NOT_SAME_AND_NOT_RAW(get_time_transformation(), value);

    UPDATE_SETTING(TimeTransformation, value);
    api_->composite.set_z_fft_shift(value == TimeTransformation::STFT);

    if (api_->compute.get_is_computation_stopped())
        return ApiCode::OK;

    api_->compute.get_compute_pipe()->request(ICS::UpdateTimeTransformationAlgorithm);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Time Tr.Freq.

void TransformApi::check_p_limits() const
{
    int upper_bound = static_cast<int>(get_time_transformation_size()) - 1;

    if (std::cmp_greater(get_p_accu_level(), upper_bound))
    {
        LOG_WARN("z width is greater than the time window, setting it: {}", upper_bound);
        set_p_accu_level(upper_bound);
    }

    upper_bound -= get_p_accu_level();

    if (get_p_index() > static_cast<uint>(upper_bound))
    {
        LOG_WARN("z start + z width is greater than the time window, setting z start to: {}", upper_bound);
        set_p_index(upper_bound);
    }
}

ApiCode TransformApi::set_p_index(uint value) const
{
    NOT_SAME_AND_NOT_RAW(get_p_index(), value);

    SET_SETTING(P, start, value);
    check_p_limits();

    api_->compute.pipe_refresh();

    return ApiCode::OK;
}

ApiCode TransformApi::set_p_accu_level(uint p_value) const
{
    NOT_SAME_AND_NOT_RAW(get_p_accu_level(), p_value);

    SET_SETTING(P, width, p_value);
    check_p_limits();

    api_->compute.pipe_refresh();

    return ApiCode::OK;
}

void TransformApi::check_q_limits() const
{
    int upper_bound = static_cast<int>(get_time_transformation_size()) - 1;

    if (std::cmp_greater(get_q_accu_level(), upper_bound))
    {
        LOG_WARN("z2 width is greater than the time window, setting it: {}", upper_bound);
        set_q_accu_level(upper_bound);
    }

    upper_bound -= get_q_accu_level();

    if (get_q_index() > static_cast<uint>(upper_bound))
    {
        LOG_WARN("z2 start + z2 width is greater than the time window, setting z2 start to: {}", upper_bound);
        set_q_index(upper_bound);
    }
}

ApiCode TransformApi::set_q_index(uint value) const
{
    NOT_SAME_AND_NOT_RAW(get_q_index(), value);

    SET_SETTING(Q, start, value);
    check_q_limits();

    api_->compute.pipe_refresh();

    return ApiCode::OK;
}

ApiCode TransformApi::set_q_accu_level(uint value) const
{
    NOT_SAME_AND_NOT_RAW(get_q_accu_level(), value);

    SET_SETTING(Q, width, value);
    check_q_limits();

    api_->compute.pipe_refresh();

    return ApiCode::OK;
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