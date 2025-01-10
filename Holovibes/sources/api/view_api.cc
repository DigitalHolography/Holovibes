#include "view_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region 3D Cuts View

bool ViewApi::set_3d_cuts_view(bool enabled) const
{
    // No 3d cuts in moments mode
    if (api_->compute.get_is_computation_stopped() || api_->input.get_data_type() == RecordedDataType::MOMENTS)
        return false;

    if (enabled)
    {
        try
        {
            api_->compute.get_compute_pipe()->request(ICS::TimeTransformationCuts);
            while (api_->compute.get_compute_pipe()->is_requested(ICS::TimeTransformationCuts))
                continue;

            api_->window_pp.set_enabled(true, WindowKind::YZview);
            api_->window_pp.set_enabled(true, WindowKind::XZview);
            set_cuts_view_enabled(true);

            api_->compute.pipe_refresh();

            return true;
        }
        catch (const std::logic_error& e)
        {
            LOG_ERROR("Catch {}", e.what());
        }
    }
    else
    {
        api_->window_pp.set_enabled(false, WindowKind::YZview);
        api_->window_pp.set_enabled(false, WindowKind::XZview);
        set_cuts_view_enabled(false);

        api_->compute.get_compute_pipe()->request(ICS::DeleteTimeTransformationCuts);
        while (api_->compute.get_compute_pipe()->is_requested(ICS::DeleteTimeTransformationCuts))
            continue;

        if (api_->record.get_record_mode() == RecordMode::CUTS_XZ ||
            api_->record.get_record_mode() == RecordMode::CUTS_YZ)
            api_->record.set_record_mode_enum(RecordMode::HOLOGRAM);

        return true;
    }

    return false;
}

#pragma endregion

#pragma region Filter2D View

void ViewApi::set_filter2d_view(bool enabled) const
{
    if (api_->compute.get_compute_mode() == Computation::Raw || api_->compute.get_is_computation_stopped())
        return;

    auto pipe = api_->compute.get_compute_pipe();
    if (enabled)
    {
        pipe->request(ICS::Filter2DView);
        while (pipe->is_requested(ICS::Filter2DView))
            continue;

        api_->contrast.set_log_enabled(true, WindowKind::Filter2D);
        api_->compute.pipe_refresh();
    }
    else
    {
        pipe->request(ICS::DisableFilter2DView);
        while (pipe->is_requested(ICS::DisableFilter2DView))
            continue;
    }
}

#pragma endregion

#pragma region Chart View

void ViewApi::set_chart_display(bool enabled) const
{
    if (get_chart_display_enabled() == enabled)
        return;

    try
    {
        auto pipe = api_->compute.get_compute_pipe();
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

#pragma region Lens View

void ViewApi::set_lens_view(bool enabled) const
{
    if (api_->compute.get_is_computation_stopped() || api_->compute.get_compute_mode() == Computation::Raw ||
        api_->input.get_data_type() == RecordedDataType::MOMENTS && enabled)
        return;

    set_lens_view_enabled(enabled);

    if (!enabled)
    {
        auto pipe = api_->compute.get_compute_pipe();
        pipe->request(ICS::DisableLensView);
        while (pipe->is_requested(ICS::DisableLensView))
            continue;
    }
}

#pragma endregion

#pragma region Raw View

void ViewApi::set_raw_view(bool enabled) const
{
    if (api_->compute.get_is_computation_stopped() || api_->compute.get_compute_mode() == Computation::Raw ||
        api_->input.get_data_type() == RecordedDataType::MOMENTS)
        return;

    if (enabled && api_->transform.get_batch_size() > api_->compute.get_output_buffer_size())
    {
        LOG_ERROR("[RAW VIEW] Batch size must be lower than output queue size");
        return;
    }

    auto pipe = api_->compute.get_compute_pipe();
    set_raw_view_enabled(enabled);

    auto request = enabled ? ICS::RawView : ICS::DisableRawView;

    pipe->request(request);
    while (pipe->is_requested(request))
        continue;

    api_->compute.pipe_refresh();
}

#pragma endregion

#pragma region Last Image

void* ViewApi::get_raw_last_image() const
{
    if (api_->compute.get_input_queue())
        return api_->compute.get_input_queue().get()->get_last_image();

    return nullptr;
}

void* ViewApi::get_hologram_last_image() const
{
    if (api_->compute.get_gpu_output_queue())
        return api_->compute.get_gpu_output_queue()->get_last_image();

    return nullptr;
}

#pragma endregion

} // namespace holovibes::api