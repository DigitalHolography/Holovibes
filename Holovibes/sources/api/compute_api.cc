#include "compute_api.hh"

#include "API.hh"

namespace holovibes::api
{

void ComputeApi::close_critical_compute()
{
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

void ComputeApi::stop_all_worker_controller() { Holovibes::instance().stop_all_worker_controller(); }

void ComputeApi::handle_update_exception()
{
    api_->transform.set_p_index(0);
    api_->transform.set_time_transformation_size(1);
    api_->global_pp.disable_convolution();
    api_->filter2d.enable_filter("");
}

#pragma region Pipe

void ComputeApi::disable_pipe_refresh()
{
    try
    {
        get_compute_pipe()->clear_request(ICS::RefreshEnabled);
    }
    catch (const std::runtime_error&)
    {
        LOG_DEBUG("Pipe not initialized: {}", e.what());
    }
}

void ComputeApi::enable_pipe_refresh()
{
    try
    {
        get_compute_pipe()->set_requested(ICS::RefreshEnabled, true);
    }
    catch (const std::runtime_error&)
    {
        LOG_DEBUG("Pipe not initialized: {}", e.what());
    }
}

void ComputeApi::pipe_refresh()
{
    if (api_->input.get_import_type() == ImportType::None)
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

void ComputeApi::create_pipe()
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

#pragma endregion

#pragma region Compute Mode

void ComputeApi::set_computation_mode(Computation mode)
{
    if (api_->input.get_data_type() == RecordedDataType::MOMENTS && mode == Computation::Raw)
        return;

    close_critical_compute();

    set_compute_mode(mode);
    create_pipe();

    if (mode == Computation::Hologram)
    {
        api_->view.change_window(WindowKind::XYview);
        api_->contrast.set_contrast_enabled(true);
    }
    else
        api_->record.set_record_mode_enum(
            RecordMode::RAW); // Force set record mode to raw because it cannot be anything else
}

#pragma endregion

#pragma region Img Type

ApiCode ComputeApi::set_view_mode(const ImgType type)
{
    if (type == get_img_type())
        return ApiCode::NO_CHANGE;

    if (api_->input.get_import_type() == ImportType::None)
        return ApiCode::NOT_STARTED;

    if (get_compute_mode() == Computation::Raw)
        return ApiCode::WRONG_MODE;

    try
    {
        bool composite = type == ImgType::Composite || get_img_type() == ImgType::Composite;

        set_img_type(type);

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

void ComputeApi::loaded_moments_data()
{
    api_->transform.set_batch_size(3);  // Moments are read in batch of 3 (since there are three moments)
    api_->transform.set_time_stride(3); // The user can change the time stride, but setting it to 3
                                        // is a good basis to analyze moments
}

#pragma endregion

} // namespace holovibes::api
