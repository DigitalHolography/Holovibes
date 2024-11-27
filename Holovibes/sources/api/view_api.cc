#include "view_api.hh"

namespace holovibes::api
{

#pragma region Open / Close Views

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

        if (get_record_mode() == RecordMode::CUTS_XZ || get_record_mode() == RecordMode::CUTS_YZ)
            set_record_mode_enum(RecordMode::HOLOGRAM);

        return true;
    }

    return false;
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
        pipe_refresh();
    }
    else
    {
        pipe->request(ICS::DisableFilter2DView);
        while (pipe->is_requested(ICS::DisableFilter2DView))
            continue;
    }
}

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

#pragma endregion

#pragma region Last Image

void* get_raw_last_image()
{
    if (get_input_queue())
        return get_input_queue().get()->get_last_image();

    return nullptr;
}

// void* get_raw_view_last_image(); // get_input_queue().get()

void* get_hologram_last_image()
{
    if (get_gpu_output_queue())
        return get_gpu_output_queue().get()->get_last_image();

    return nullptr;
}

// void* get_lens_last_image();     // api::get_compute_pipe()->get_lens_queue().get()
// void* get_xz_last_image();       // api::get_compute_pipe()->get_stft_slice_queue(0).get()
// void* get_yz_last_image();       // api::get_compute_pipe()->get_stft_slice_queue(1).get()
// void* get_filter2d_last_image(); // api::get_compute_pipe()->get_filter2d_view_queue().get()
// void* get_chart_last_image();    // api::get_compute_pipe()->get_chart_display_queue().get()

#pragma endregion

} // namespace holovibes::api