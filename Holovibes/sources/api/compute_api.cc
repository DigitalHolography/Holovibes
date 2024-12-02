#include "compute_api.hh"

namespace holovibes::api
{

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

void handle_update_exception()
{
    api::set_p_index(0);
    api::set_time_transformation_size(1);
    api::disable_convolution();
    api::enable_filter("");
}

#pragma region Pipe

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

#pragma endregion

#pragma region Change Mode

void set_computation_mode(Computation mode)
{
    if (get_data_type() == RecordedDataType::MOMENTS && mode == Computation::Raw)
        return;

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

void loaded_moments_data()
{
    set_batch_size(3);  // Cannot call the api.cc function because some parameters are not defined yet.
    set_time_stride(3); // The user can change the time stride, but setting it to 3
                        // is a good basis to analyze moments

    // There are plenty of settings not used in data type moments but not modified here;
    // it's because these settings' value have no influence at that point (ex: space/time transforms).
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

} // namespace holovibes::api
