#include "compute_api.hh"

#include "API.hh"

namespace holovibes::api
{

void ComputeApi::close_critical_compute() const
{
    if (get_is_computation_stopped())
        return;

    if (api_->global_pp.get_convolution_enabled())
        api_->global_pp.disable_convolution();

    if (api_->view.get_cuts_view_enabled())
        api_->view.set_3d_cuts_view(false);

    if (api_->view.get_filter2d_view_enabled())
        api_->view.set_filter2d_view(false);

    if (api_->view.get_lens_view_enabled())
        api_->view.set_lens_view(false);

    if (api_->view.get_raw_view_enabled())
        api_->view.set_raw_view(false);

    Holovibes::instance().stop_compute();
}

void ComputeApi::stop_all_worker_controller() const { Holovibes::instance().stop_all_worker_controller(); }

ApiCode ComputeApi::start() const
{
    if (api_->input.get_import_type() == ImportType::None)
        return ApiCode::NO_IN_DATA;

    // Stop any computation currently running and file reading
    if (!get_is_computation_stopped())
    {
        close_critical_compute();
        Holovibes::instance().stop_frame_read();
    }

    Holovibes::instance().init_pipe();
    Holovibes::instance().start_compute_worker();
    set_is_computation_stopped(false);
    pipe_refresh();

    if (api_->input.get_import_type() == ImportType::Camera)
        Holovibes::instance().start_camera_frame_read();
    else
        Holovibes::instance().start_file_frame_read();

    return ApiCode::OK;
}

#pragma region Pipe

void ComputeApi::disable_pipe_refresh() const
{
    if (get_is_computation_stopped())
        return;

    try
    {
        get_compute_pipe()->clear_request(ICS::RefreshEnabled);
    }
    catch (const std::runtime_error&)
    {
        LOG_DEBUG("Pipe not initialized: {}", e.what());
    }
}

void ComputeApi::enable_pipe_refresh() const
{
    if (get_is_computation_stopped())
        return;

    try
    {
        get_compute_pipe()->set_requested(ICS::RefreshEnabled, true);
    }
    catch (const std::runtime_error&)
    {
        LOG_DEBUG("Pipe not initialized: {}", e.what());
    }
}

void ComputeApi::pipe_refresh() const
{
    if (get_is_computation_stopped())
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

#pragma endregion

#pragma region Compute Mode

ApiCode ComputeApi::set_computation_mode(Computation mode) const
{
    if (mode == get_compute_mode())
        return ApiCode::NO_CHANGE;

    if (get_is_computation_stopped())
    {
        UPDATE_SETTING(ComputeMode, mode);
        return ApiCode::OK;
    }

    set_compute_mode(mode);

    if (mode == Computation::Hologram)
    {
        api_->view.change_window(WindowKind::XYview);
        api_->contrast.set_contrast_enabled(true);
    }
    else
        api_->record.set_record_mode_enum(
            RecordMode::RAW); // Force set record mode to raw because it cannot be anything else

    get_compute_pipe()->request(ICS::OutputBuffer);
    while (get_compute_pipe()->is_requested(ICS::OutputBuffer))
        continue;

    return ApiCode::OK;
}

#pragma endregion

#pragma region Img Type

ApiCode ComputeApi::set_img_type(const ImgType type) const
{
    if (type == get_img_type())
        return ApiCode::NO_CHANGE;

    if (get_compute_mode() == Computation::Raw)
        return ApiCode::WRONG_COMP_MODE;

    if (get_is_computation_stopped())
    {
        UPDATE_SETTING(ImageType, type);
        return ApiCode::OK;
    }

    try
    {
        bool composite = type == ImgType::Composite || get_img_type() == ImgType::Composite;

        UPDATE_SETTING(ImageType, type);

        // Switching to composite or back from composite needs a recreation of the pipe since buffers size will be *3
        if (composite)
            set_compute_mode(Computation::Hologram);
        else
            pipe_refresh();
    }
    catch (const std::runtime_error&) // The pipe is not initialized
    {
        return ApiCode::FAILURE;
    }

    return ApiCode::OK;
}

void ComputeApi::loaded_moments_data() const
{
    api_->transform.set_batch_size(3);  // Moments are read in batch of 3 (since there are three moments)
    api_->transform.set_time_stride(3); // The user can change the time stride, but setting it to 3
                                        // is a good basis to analyze moments
}

#pragma endregion

} // namespace holovibes::api
