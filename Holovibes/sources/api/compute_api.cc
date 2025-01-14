#include "compute_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Compute

ApiCode ComputeApi::stop() const
{
    if (get_is_computation_stopped())
        return ApiCode::NOT_STARTED;

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
    API.information.stop_benchmark();
    set_is_computation_stopped(true);

    Holovibes::instance().stop_frame_read();

    return ApiCode::OK;
}

ApiCode ComputeApi::start() const
{
    if (api_->input.get_import_type() == ImportType::None)
        return ApiCode::NO_IN_DATA;

    // Stop any computation currently running and file reading
    stop();

    // Create the pipe and start the pipe
    Holovibes::instance().start_compute();
    set_is_computation_stopped(false);

    if (api_->global_pp.get_convolution_enabled())
        api_->global_pp.enable_convolution(api_->global_pp.get_convolution_file_name());
    if (api_->filter2d.get_filter2d_enabled() && !api_->filter2d.get_filter_file_name().empty())
        api_->filter2d.enable_filter(api_->filter2d.get_filter_file_name());
    else
        pipe_refresh();

    if (api_->input.get_import_type() == ImportType::Camera)
        Holovibes::instance().start_camera_frame_read();
    else
        Holovibes::instance().start_file_frame_read();

    API.information.start_benchmark();

    return ApiCode::OK;
}

#pragma endregion

#pragma region Pipe

ApiCode ComputeApi::disable_pipe_refresh() const
{
    if (get_is_computation_stopped())
        return ApiCode::NOT_STARTED;

    get_compute_pipe()->clear_request(ICS::RefreshEnabled);

    return ApiCode::OK;
}

ApiCode ComputeApi::enable_pipe_refresh() const
{
    if (get_is_computation_stopped())
        return ApiCode::NOT_STARTED;

    get_compute_pipe()->set_requested(ICS::RefreshEnabled, true);

    return ApiCode::OK;
}

ApiCode ComputeApi::pipe_refresh() const
{
    if (get_is_computation_stopped())
        return ApiCode::NOT_STARTED;

    LOG_TRACE("pipe_refresh");
    get_compute_pipe()->request_refresh();

    return ApiCode::OK;
}

#pragma endregion

#pragma region Compute Mode

ApiCode ComputeApi::set_compute_mode(Computation mode) const
{
    if (mode == get_compute_mode())
        return ApiCode::NO_CHANGE;

    UPDATE_SETTING(ComputeMode, mode);

    if (mode == Computation::Hologram)
    {
        api_->view.change_window(WindowKind::XYview);
        api_->contrast.set_contrast_enabled(true);
    }
    else
        api_->record.set_record_mode(
            RecordMode::RAW); // Force set record mode to raw because it cannot be anything else

    if (get_is_computation_stopped())
        return ApiCode::OK;

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
            start();
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

} // namespace holovibes::api
