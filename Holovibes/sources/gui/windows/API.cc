#include "api.hh"

#include <optional>
namespace holovibes::api
{

std::optional<::holovibes::io_files::InputFrameFile*> import_file(const std::string& filename)
{
    LOG_INFO;

    if (!filename.empty())
    {

        // Will throw if the file format (extension) cannot be handled
        auto input_file = ::holovibes::io_files::InputFrameFileFactory::open(filename);

        return input_file;
    }

    return std::nullopt;
}

bool init_holovibes_import_mode(UserInterfaceDescriptor& ui_descriptor,
                                std::string& file_path,
                                unsigned int fps,
                                size_t first_frame,
                                bool load_file_in_gpu,
                                size_t last_frame)
{
    LOG_INFO;

    // Set the image rendering ui params
    ui_descriptor.holovibes_.get_cd().time_transformation_stride = std::ceil(static_cast<float>(fps) / 20.0f);
    ui_descriptor.holovibes_.get_cd().batch_size = 1;

    // Because we are in import mode
    ui_descriptor.is_enabled_camera_ = false;

    try
    {

        ui_descriptor.holovibes_.init_input_queue(ui_descriptor.file_fd_);
        ui_descriptor.holovibes_.start_file_frame_read(file_path,
                                                       true,
                                                       fps,
                                                       first_frame - 1,
                                                       last_frame - first_frame + 1,
                                                       load_file_in_gpu,
                                                       [=]() { return; });
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        ui_descriptor.is_enabled_camera_ = false;
        ui_descriptor.holovibes_.stop_compute();
        ui_descriptor.holovibes_.stop_frame_read();
        return false;
    }
    ui_descriptor.is_enabled_camera_ = true;
    return true;
}

bool import_start(UserInterfaceDescriptor& ui_descriptor,
                  std::string& file_path,
                  unsigned int fps,
                  size_t first_frame,
                  bool load_file_in_gpu,
                  size_t last_frame)
{
    LOG_INFO;

    if (!ui_descriptor.holovibes_.get_cd().is_computation_stopped)
        // if computation is running
        import_stop(ui_descriptor);

    ui_descriptor.holovibes_.get_cd().is_computation_stopped = false;
    // Gather all the usefull data from the ui import panel
    return init_holovibes_import_mode(ui_descriptor, file_path, fps, first_frame, load_file_in_gpu, last_frame);
}

void import_stop(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    ui_descriptor.holovibes_.stop_all_worker_controller();
    ui_descriptor.holovibes_.start_information_display(false);

    close_critical_compute(ui_descriptor);

    // FIXME: import_stop() and camera_none() call same methods
    // FIXME: camera_none() weird call because we are dealing with imported file
    camera_none(ui_descriptor);

    ui_descriptor.holovibes_.get_cd().is_computation_stopped = true;
}

void camera_none(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    close_critical_compute(ui_descriptor);
    if (!is_raw_mode(ui_descriptor.holovibes_))
        ui_descriptor.holovibes_.stop_compute();
    ui_descriptor.holovibes_.stop_frame_read();
    remove_infos();

    ui_descriptor.is_enabled_camera_ = false;
    ui_descriptor.holovibes_.get_cd().is_computation_stopped = true;
}

void close_critical_compute(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (ui_descriptor.holovibes_.get_cd().convolution_enabled)
        set_convolution_mode(ui_descriptor, false);

    if (ui_descriptor.holovibes_.get_cd().time_transformation_cuts_enabled)
        cancel_time_transformation_cuts(ui_descriptor.holovibes_, []() { return; });

    ui_descriptor.holovibes_.stop_compute();
}

bool is_raw_mode(Holovibes& holovibes)
{
    LOG_INFO;
    return holovibes.get_cd().compute_mode == Computation::Raw;
}

void remove_infos()
{
    LOG_INFO;
    Holovibes::instance().get_info_container().clear();
}

void set_convolution_mode(UserInterfaceDescriptor& ui_descriptor, const bool value)
{
    LOG_INFO;

    try
    {
        auto pipe = ui_descriptor.holovibes_.get_compute_pipe();

        if (value)
        {
            pipe->request_convolution();
            // Wait for the convolution to be enabled for notify
            while (pipe->get_convolution_requested())
                continue;
        }
        else
        {
            pipe->request_disable_convolution();
            // Wait for the convolution to be disabled for notify
            while (pipe->get_disable_convolution_requested())
                continue;
        }
    }
    catch (const std::exception& e)
    {
        ui_descriptor.holovibes_.get_cd().convolution_enabled = false;
        LOG_ERROR << e.what();
    }
}

void cancel_time_transformation_cuts(Holovibes& holovibes, std::function<void()> callback)
{
    LOG_INFO;
    if (holovibes.get_cd().time_transformation_cuts_enabled)
    {

        holovibes.get_cd().contrast_max_slice_xz = false;
        holovibes.get_cd().contrast_max_slice_yz = false;
        holovibes.get_cd().log_scale_slice_xz_enabled = false;
        holovibes.get_cd().log_scale_slice_yz_enabled = false;
        holovibes.get_cd().img_acc_slice_xz_enabled = false;
        holovibes.get_cd().img_acc_slice_yz_enabled = false;

        holovibes.get_compute_pipe().get()->insert_fn_end_vect(callback);

        try
        {
            // Wait for refresh to be enabled for notify
            while (holovibes.get_compute_pipe()->get_refresh_request())
                continue;
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what();
        }

        holovibes.get_cd().time_transformation_cuts_enabled = false;
    }
}

// Check that value is higher or equal than 0
void set_record_frame_step(unsigned int& record_frame_step, int value) { record_frame_step = value; }

bool start_record_preconditions(const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const RecordMode record_mode,
                                std::string& output_path,
                                std::string& batch_input_path)
{
    LOG_INFO;
    // Preconditions to start record

    if (!nb_frame_checked)
        nb_frames_to_record = std::nullopt;

    if ((batch_enabled || record_mode == RecordMode::CHART) && nb_frames_to_record == std::nullopt)
    {
        LOG_ERROR << "Number of frames must be activated";
        return false;
    }

    if (batch_enabled && batch_input_path.empty())
    {
        LOG_ERROR << "No batch input file";
        return false;
    }

    return true;
}

void start_record(Holovibes& holovibes,
                  const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  const RecordMode record_mode,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback)
{
    LOG_INFO;

    if (batch_enabled)
    {
        holovibes.start_batch_gpib(batch_input_path, output_path, nb_frames_to_record.value(), record_mode, callback);
    }
    else
    {
        if (record_mode == RecordMode::CHART)
        {
            holovibes.start_chart_record(output_path, nb_frames_to_record.value(), callback);
        }
        else if (record_mode == RecordMode::HOLOGRAM)
        {
            holovibes.start_frame_record(output_path, nb_frames_to_record, false, 0, callback);
        }
        else if (record_mode == RecordMode::RAW)
        {
            holovibes.start_frame_record(output_path, nb_frames_to_record, true, 0, callback);
        }
    }
}

void stop_record(Holovibes& holovibes, const RecordMode record_mode)
{
    LOG_INFO;
    holovibes.stop_batch_gpib();

    if (record_mode == RecordMode::CHART)
        holovibes.stop_chart_record();
    else if (record_mode == RecordMode::HOLOGRAM || record_mode == RecordMode::RAW)
        holovibes.stop_frame_record();
}

const std::string browse_record_output_file(std::string& std_filepath,
                                            std::string& record_output_directory,
                                            std::string& default_output_filename)
{
    LOG_INFO;

    // FIXME: path separator should depend from system
    std::replace(std_filepath.begin(), std_filepath.end(), '/', '\\');
    std::filesystem::path path = std::filesystem::path(std_filepath);

    // FIXME Opti: we could be all these 3 operations below on a single string processing
    record_output_directory = path.parent_path().string();
    const std::string file_ext = path.extension().string();
    default_output_filename = path.stem().string();

    return file_ext;
}

void set_record_mode(const std::string& text, RecordMode& record_mode)
{
    LOG_INFO;

    if (text == "Chart")
        record_mode = RecordMode::CHART;
    else if (text == "Processed Image")
        record_mode = RecordMode::HOLOGRAM;
    else if (text == "Raw Image")
        record_mode = RecordMode::RAW;
    else
        throw std::exception("Record mode not handled");
}

void close_windows(Holovibes& holovibes,
                   std::unique_ptr<::holovibes::gui::RawWindow>& mainDisplay,
                   std::unique_ptr<::holovibes::gui::SliceWindow>& sliceXZ,
                   std::unique_ptr<::holovibes::gui::SliceWindow>& sliceYZ,
                   std::unique_ptr<::holovibes::gui::RawWindow>& lens_window,
                   std::unique_ptr<::holovibes::gui::RawWindow>& raw_window,
                   std::unique_ptr<::holovibes::gui::Filter2DWindow>& filter2d_window,
                   std::unique_ptr<::holovibes::gui::PlotWindow>& plot_window_)
{
    LOG_INFO;
    sliceXZ.reset(nullptr);
    sliceYZ.reset(nullptr);

    plot_window_.reset(nullptr);
    mainDisplay.reset(nullptr);

    lens_window.reset(nullptr);
    holovibes.get_cd().gpu_lens_display_enabled = false;

    filter2d_window.reset(nullptr);
    holovibes.get_cd().filter2d_view_enabled = false;

    /* Raw view & recording */
    raw_window.reset(nullptr);
    holovibes.get_cd().raw_view_enabled = false;

    // Disable overlays
    holovibes.get_cd().reticle_enabled = false;
}

void set_computation_mode(Holovibes& holovibes, const uint image_mode_index)
{
    LOG_INFO;
    if (image_mode_index == 0)
    {
        holovibes.get_cd().compute_mode = Computation::Raw;
    }
    else if (image_mode_index == 1)
    {
        holovibes.get_cd().compute_mode = Computation::Hologram;
    }
}

void set_camera_timeout()
{
    LOG_INFO;
    camera::FRAME_TIMEOUT = global::global_config.frame_timeout;
}

void change_camera(::holovibes::gui::MainWindow& mainwindow,
                   Holovibes& holovibes,
                   CameraKind c,
                   CameraKind& kCamera,
                   bool& is_enabled_camera,
                   ::holovibes::UserInterfaceDescriptor::ImportType& import_type,
                   std::unique_ptr<::holovibes::gui::RawWindow>& mainDisplay,
                   const uint image_mode_index)
{
    LOG_INFO;

    mainDisplay.reset(nullptr);
    if (!is_raw_mode(holovibes))
        holovibes.stop_compute();
    holovibes.stop_frame_read();

    set_camera_timeout();

    set_computation_mode(holovibes, image_mode_index);

    holovibes.start_camera_frame_read(c);
    is_enabled_camera = true;
    set_image_mode(mainwindow, holovibes, nullptr, image_mode_index);
    import_type = ::holovibes::UserInterfaceDescriptor::ImportType::Camera;
    kCamera = c;

    holovibes.get_cd().is_computation_stopped = false;
}

void set_image_mode(::holovibes::gui::MainWindow& mainwindow,
                    Holovibes& holovibes,
                    const bool is_null_mode,
                    const uint image_mode_index)
{
    LOG_INFO;
    if (!is_null_mode)
    {
        // Call comes from ui
        if (image_mode_index == 0)
            mainwindow.set_raw_mode();
        else
            mainwindow.set_holographic_mode();
    }
    else if (holovibes.get_cd().compute_mode == Computation::Raw)
        mainwindow.set_raw_mode();
    else if (holovibes.get_cd().compute_mode == Computation::Hologram)
        mainwindow.set_holographic_mode();
}

} // namespace holovibes::api