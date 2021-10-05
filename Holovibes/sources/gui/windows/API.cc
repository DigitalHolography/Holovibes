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
    if (!is_raw_mode(ui_descriptor))
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
        unset_convolution_mode(ui_descriptor);

    if (ui_descriptor.holovibes_.get_cd().time_transformation_cuts_enabled)
        cancel_time_transformation_cuts(ui_descriptor, []() { return; });

    ui_descriptor.holovibes_.stop_compute();
}

bool is_raw_mode(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    return ui_descriptor.holovibes_.get_cd().compute_mode == Computation::Raw;
}

void remove_infos()
{
    LOG_INFO;
    Holovibes::instance().get_info_container().clear();
}

void set_convolution_mode(UserInterfaceDescriptor& ui_descriptor, std::string& str)
{
    LOG_INFO;

    ui_descriptor.holovibes_.get_cd().set_convolution(true, str);

    try
    {
        auto pipe = ui_descriptor.holovibes_.get_compute_pipe();

        pipe->request_convolution();
        // Wait for the convolution to be enabled for notify
        while (pipe->get_convolution_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        ui_descriptor.holovibes_.get_cd().convolution_enabled = false;
        LOG_ERROR << e.what();
    }
}

void unset_convolution_mode(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    try
    {
        auto pipe = ui_descriptor.holovibes_.get_compute_pipe();

        pipe->request_disable_convolution();
        // Wait for the convolution to be disabled for notify
        while (pipe->get_disable_convolution_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        ui_descriptor.holovibes_.get_cd().convolution_enabled = false;
        LOG_ERROR << e.what();
    }
}

bool cancel_time_transformation_cuts(UserInterfaceDescriptor& ui_descriptor, std::function<void()> callback)
{
    LOG_INFO;

    if (!ui_descriptor.holovibes_.get_cd().time_transformation_cuts_enabled)
    {
        return false;
    }

    ui_descriptor.holovibes_.get_cd().contrast_max_slice_xz = false;
    ui_descriptor.holovibes_.get_cd().contrast_max_slice_yz = false;
    ui_descriptor.holovibes_.get_cd().log_scale_slice_xz_enabled = false;
    ui_descriptor.holovibes_.get_cd().log_scale_slice_yz_enabled = false;
    ui_descriptor.holovibes_.get_cd().img_acc_slice_xz_enabled = false;
    ui_descriptor.holovibes_.get_cd().img_acc_slice_yz_enabled = false;

    ui_descriptor.holovibes_.get_compute_pipe().get()->insert_fn_end_vect(callback);

    try
    {
        // Wait for refresh to be enabled for notify
        while (ui_descriptor.holovibes_.get_compute_pipe()->get_refresh_request())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
    }

    ui_descriptor.holovibes_.get_cd().time_transformation_cuts_enabled = false;

    ui_descriptor.sliceXZ.reset(nullptr);
    ui_descriptor.sliceYZ.reset(nullptr);

    if (ui_descriptor.mainDisplay)
    {
        ui_descriptor.mainDisplay->setCursor(Qt::ArrowCursor);
        ui_descriptor.mainDisplay->getOverlayManager().disable_all(::holovibes::gui::SliceCross);
        ui_descriptor.mainDisplay->getOverlayManager().disable_all(::holovibes::gui::Cross);
    }

    return true;
}

// Check that value is higher or equal than 0
void set_record_frame_step(UserInterfaceDescriptor& ui_descriptor, int value)
{
    ui_descriptor.record_frame_step_ = value;
}

bool start_record_preconditions(const UserInterfaceDescriptor& ui_descriptor,
                                const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const std::string& batch_input_path)
{
    LOG_INFO;
    // Preconditions to start record

    if (!nb_frame_checked)
        nb_frames_to_record = std::nullopt;

    if ((batch_enabled || ui_descriptor.record_mode_ == RecordMode::CHART) && nb_frames_to_record == std::nullopt)
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

void start_record(UserInterfaceDescriptor& ui_descriptor,
                  const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback)
{
    LOG_INFO;

    if (batch_enabled)
    {
        ui_descriptor.holovibes_.start_batch_gpib(batch_input_path,
                                                  output_path,
                                                  nb_frames_to_record.value(),
                                                  ui_descriptor.record_mode_,
                                                  callback);
    }
    else
    {
        if (ui_descriptor.record_mode_ == RecordMode::CHART)
        {
            ui_descriptor.holovibes_.start_chart_record(output_path, nb_frames_to_record.value(), callback);
        }
        else if (ui_descriptor.record_mode_ == RecordMode::HOLOGRAM)
        {
            ui_descriptor.holovibes_.start_frame_record(output_path, nb_frames_to_record, false, 0, callback);
        }
        else if (ui_descriptor.record_mode_ == RecordMode::RAW)
        {
            ui_descriptor.holovibes_.start_frame_record(output_path, nb_frames_to_record, true, 0, callback);
        }
    }
}

void stop_record(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    ui_descriptor.holovibes_.stop_batch_gpib();

    if (ui_descriptor.record_mode_ == RecordMode::CHART)
        ui_descriptor.holovibes_.stop_chart_record();
    else if (ui_descriptor.record_mode_ == RecordMode::HOLOGRAM || ui_descriptor.record_mode_ == RecordMode::RAW)
        ui_descriptor.holovibes_.stop_frame_record();
}

const std::string browse_record_output_file(UserInterfaceDescriptor& ui_descriptor, std::string& std_filepath)
{
    LOG_INFO;

    // FIXME: path separator should depend from system
    std::replace(std_filepath.begin(), std_filepath.end(), '/', '\\');
    std::filesystem::path path = std::filesystem::path(std_filepath);

    // FIXME Opti: we could be all these 3 operations below on a single string processing
    ui_descriptor.record_output_directory_ = path.parent_path().string();
    const std::string file_ext = path.extension().string();
    ui_descriptor.default_output_filename_ = path.stem().string();

    return file_ext;
}

void set_record_mode(UserInterfaceDescriptor& ui_descriptor, const std::string& text)
{
    LOG_INFO;

    if (text == "Chart")
        ui_descriptor.record_mode_ = RecordMode::CHART;
    else if (text == "Processed Image")
        ui_descriptor.record_mode_ = RecordMode::HOLOGRAM;
    else if (text == "Raw Image")
        ui_descriptor.record_mode_ = RecordMode::RAW;
    else
        throw std::exception("Record mode not handled");
}

void close_windows(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    ui_descriptor.sliceXZ.reset(nullptr);
    ui_descriptor.sliceYZ.reset(nullptr);

    ui_descriptor.plot_window_.reset(nullptr);
    ui_descriptor.mainDisplay.reset(nullptr);

    ui_descriptor.lens_window.reset(nullptr);
    ui_descriptor.holovibes_.get_cd().gpu_lens_display_enabled = false;

    ui_descriptor.filter2d_window.reset(nullptr);
    ui_descriptor.holovibes_.get_cd().filter2d_view_enabled = false;

    /* Raw view & recording */
    ui_descriptor.raw_window.reset(nullptr);
    ui_descriptor.holovibes_.get_cd().raw_view_enabled = false;

    // Disable overlays
    ui_descriptor.holovibes_.get_cd().reticle_enabled = false;
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
                   UserInterfaceDescriptor& ui_descriptor,
                   CameraKind c,
                   const uint image_mode_index)
{
    LOG_INFO;

    ui_descriptor.mainDisplay.reset(nullptr);
    if (!is_raw_mode(ui_descriptor))
        ui_descriptor.holovibes_.stop_compute();
    ui_descriptor.holovibes_.stop_frame_read();

    set_camera_timeout();

    set_computation_mode(ui_descriptor.holovibes_, image_mode_index);

    ui_descriptor.holovibes_.start_camera_frame_read(c);
    ui_descriptor.is_enabled_camera_ = true;
    set_image_mode(mainwindow, ui_descriptor, true, image_mode_index);
    ui_descriptor.import_type_ = ::holovibes::UserInterfaceDescriptor::ImportType::Camera;
    ui_descriptor.kCamera = c;

    ui_descriptor.holovibes_.get_cd().is_computation_stopped = false;
}

void set_image_mode(::holovibes::gui::MainWindow& mainwindow,
                    UserInterfaceDescriptor& ui_descriptor,
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
    else if (ui_descriptor.holovibes_.get_cd().compute_mode == Computation::Raw)
        mainwindow.set_raw_mode();
    else if (ui_descriptor.holovibes_.get_cd().compute_mode == Computation::Hologram)
        mainwindow.set_holographic_mode();
}

void pipe_refresh(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return;

    try
    {
        // FIXME: Should better not use a if structure with 2 method access, 1 dereferencing, and 1 negation bitwise
        // operation to set a boolean
        // But maybe a simple read access that create a false condition result is better than simply making a
        // writting access
        if (!ui_descriptor.holovibes_.get_compute_pipe()->get_request_refresh())
            ui_descriptor.holovibes_.get_compute_pipe()->request_refresh();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what();
    }
}

void set_p_accu(UserInterfaceDescriptor& ui_descriptor, bool is_p_accu, uint p_value)
{
    ui_descriptor.holovibes_.get_cd().p_accu_enabled = is_p_accu;
    ui_descriptor.holovibes_.get_cd().p_acc_level = p_value;
    pipe_refresh(ui_descriptor);
}

void set_x_accu(UserInterfaceDescriptor& ui_descriptor, bool is_x_accu, uint x_value)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().x_accu_enabled = is_x_accu;
    ui_descriptor.holovibes_.get_cd().x_acc_level = x_value;
    pipe_refresh(ui_descriptor);
}

void set_y_accu(UserInterfaceDescriptor& ui_descriptor, bool is_y_accu, uint y_value)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().y_accu_enabled = is_y_accu;
    ui_descriptor.holovibes_.get_cd().y_acc_level = y_value;
    pipe_refresh(ui_descriptor);
}

void set_q_accu(UserInterfaceDescriptor& ui_descriptor, bool is_q_accu, uint q_value)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().q_acc_enabled = is_q_accu;
    ui_descriptor.holovibes_.get_cd().q_acc_level = q_value;
    pipe_refresh(ui_descriptor);
}

void set_x_y(UserInterfaceDescriptor& ui_descriptor, const camera::FrameDescriptor& frame_descriptor, uint x, uint y)
{
    LOG_INFO;

    if (x < frame_descriptor.width)
        ui_descriptor.holovibes_.get_cd().x_cuts = x;

    if (y < frame_descriptor.height)
        ui_descriptor.holovibes_.get_cd().y_cuts = y;
}

const bool set_p(UserInterfaceDescriptor& ui_descriptor, int value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    if (value < static_cast<int>(ui_descriptor.holovibes_.get_cd().time_transformation_size))
    {
        ui_descriptor.holovibes_.get_cd().pindex = value;
        pipe_refresh(ui_descriptor);
        return true;
    }
    else
        LOG_ERROR << "p param has to be between 1 and #img";
    return false;
}

void set_q(UserInterfaceDescriptor& ui_descriptor, int value)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().q_index = value;
}

void set_composite_intervals(UserInterfaceDescriptor& ui_descriptor, uint composite_p_red, uint composite_p_blue)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_p_red = composite_p_red;
    ui_descriptor.holovibes_.get_cd().composite_p_blue = composite_p_blue;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_h_min(UserInterfaceDescriptor& ui_descriptor, uint composite_p_min_h)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_p_min_h = composite_p_min_h;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_h_max(UserInterfaceDescriptor& ui_descriptor, uint composite_p_max_h)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_p_max_h = composite_p_max_h;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_s_min(UserInterfaceDescriptor& ui_descriptor, uint composite_p_min_s)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_p_min_s = composite_p_min_s;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_s_max(UserInterfaceDescriptor& ui_descriptor, uint composite_p_max_s)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_p_max_s = composite_p_max_s;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_v_min(UserInterfaceDescriptor& ui_descriptor, uint composite_p_min_v)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_p_min_v = composite_p_min_v;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_v_max(UserInterfaceDescriptor& ui_descriptor, uint composite_p_max_v)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_p_max_v = composite_p_max_v;
    pipe_refresh(ui_descriptor);
}

void set_composite_weights(UserInterfaceDescriptor& ui_descriptor, uint weight_r, uint weight_g, uint weight_b)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().weight_r = weight_r;
    ui_descriptor.holovibes_.get_cd().weight_g = weight_g;
    ui_descriptor.holovibes_.get_cd().weight_b = weight_b;
    pipe_refresh(ui_descriptor);
}

void set_composite_auto_weights(::holovibes::gui::MainWindow& mainwindow,
                                UserInterfaceDescriptor& ui_descriptor,
                                bool value)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_auto_weights_ = value;
    mainwindow.set_auto_contrast();
}

void select_composite_rgb(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    ui_descriptor.holovibes_.get_cd().composite_kind = CompositeKind::RGB;
}

void select_composite_hsv(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    ui_descriptor.holovibes_.get_cd().composite_kind = CompositeKind::HSV;
}

void actualize_frequency_channel_s(UserInterfaceDescriptor& ui_descriptor, bool composite_p_activated_s)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_p_activated_s = composite_p_activated_s;
}

void actualize_frequency_channel_v(UserInterfaceDescriptor& ui_descriptor, bool composite_p_activated_v)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().composite_p_activated_v = composite_p_activated_v;
}

void actualize_selection_h_gaussian_blur(UserInterfaceDescriptor& ui_descriptor, bool h_blur_activated)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().h_blur_activated = h_blur_activated;
}

void actualize_kernel_size_blur(UserInterfaceDescriptor& ui_descriptor, uint h_blur_kernel_size)
{
    LOG_INFO;
    ui_descriptor.holovibes_.get_cd().h_blur_kernel_size = h_blur_kernel_size;
}

bool increment_p(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    if (ui_descriptor.holovibes_.get_cd().pindex < ui_descriptor.holovibes_.get_cd().time_transformation_size)
    {
        ui_descriptor.holovibes_.get_cd().pindex++;
        mainwindow.set_auto_contrast();
        return true;
    }

    LOG_ERROR << "p param has to be between 1 and #img";
    return false;
}

bool decrement_p(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    if (ui_descriptor.holovibes_.get_cd().pindex > 0)
    {
        ui_descriptor.holovibes_.get_cd().pindex--;
        mainwindow.set_auto_contrast();
        return true;
    }

    LOG_ERROR << "p param has to be between 1 and #img";
    return false;
}

bool set_wavelength(UserInterfaceDescriptor& ui_descriptor, const double value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    ui_descriptor.holovibes_.get_cd().lambda = static_cast<float>(value) * 1.0e-9f;
    pipe_refresh(ui_descriptor);
    return true;
}

bool set_z(UserInterfaceDescriptor& ui_descriptor, const double value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    ui_descriptor.holovibes_.get_cd().zdistance = static_cast<float>(value);
    pipe_refresh(ui_descriptor);
    return true;
}

bool increment_z(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    set_z(ui_descriptor, ui_descriptor.holovibes_.get_cd().zdistance + ui_descriptor.z_step_);
    return true;
}

bool decrement_z(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    set_z(ui_descriptor, ui_descriptor.holovibes_.get_cd().zdistance - ui_descriptor.z_step_);
    return true;
}

void set_z_step(UserInterfaceDescriptor& ui_descriptor, const double value)
{
    LOG_INFO;
    ui_descriptor.z_step_ = value;
}

bool set_space_transformation(::holovibes::gui::MainWindow& mainwindow,
                              UserInterfaceDescriptor& ui_descriptor,
                              const std::string& value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    if (value == "None")
        ui_descriptor.holovibes_.get_cd().space_transformation = SpaceTransformation::None;
    else if (value == "1FFT")
        ui_descriptor.holovibes_.get_cd().space_transformation = SpaceTransformation::FFT1;
    else if (value == "2FFT")
        ui_descriptor.holovibes_.get_cd().space_transformation = SpaceTransformation::FFT2;
    else
    {
        // Shouldn't happen
        ui_descriptor.holovibes_.get_cd().space_transformation = SpaceTransformation::None;
        LOG_ERROR << "Unknown space transform: " << value << ", falling back to None";
    }

    mainwindow.set_holographic_mode();
    return true;
}

bool set_time_transformation(::holovibes::gui::MainWindow& mainwindow,
                             UserInterfaceDescriptor& ui_descriptor,
                             const std::string& value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    if (value == "STFT")
        ui_descriptor.holovibes_.get_cd().time_transformation = TimeTransformation::STFT;
    else if (value == "PCA")
        ui_descriptor.holovibes_.get_cd().time_transformation = TimeTransformation::PCA;
    else if (value == "None")
        ui_descriptor.holovibes_.get_cd().time_transformation = TimeTransformation::NONE;
    else if (value == "SSA_STFT")
        ui_descriptor.holovibes_.get_cd().time_transformation = TimeTransformation::SSA_STFT;

    mainwindow.set_holographic_mode();
    return true;
}

bool set_unwrapping_2d(UserInterfaceDescriptor& ui_descriptor, const bool value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    ui_descriptor.holovibes_.get_compute_pipe()->request_unwrapping_2d(value);
    pipe_refresh(ui_descriptor);
    return true;
}

bool set_accumulation(UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    ui_descriptor.holovibes_.get_cd().set_accumulation(ui_descriptor.holovibes_.get_cd().current_window, value);
    pipe_refresh(ui_descriptor);
    return true;
}

bool set_accumulation_level(UserInterfaceDescriptor& ui_descriptor, int value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    ui_descriptor.holovibes_.get_cd().set_accumulation_level(ui_descriptor.holovibes_.get_cd().current_window, value);
    pipe_refresh(ui_descriptor);
    return true;
}

void set_composite_area(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    ui_descriptor.mainDisplay->getOverlayManager().create_overlay<::holovibes::gui::CompositeArea>();
}

void rotateTexture(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    const WindowKind curWin = ui_descriptor.holovibes_.get_cd().current_window;

    if (curWin == WindowKind::XYview)
    {
        ui_descriptor.displayAngle = (ui_descriptor.displayAngle == 270.f) ? 0.f : ui_descriptor.displayAngle + 90.f;
        ui_descriptor.mainDisplay->setAngle(ui_descriptor.displayAngle);
    }
    else if (ui_descriptor.sliceXZ && curWin == WindowKind::XZview)
    {
        ui_descriptor.xzAngle = (ui_descriptor.xzAngle == 270.f) ? 0.f : ui_descriptor.xzAngle + 90.f;
        ui_descriptor.sliceXZ->setAngle(ui_descriptor.xzAngle);
    }
    else if (ui_descriptor.sliceYZ && curWin == WindowKind::YZview)
    {
        ui_descriptor.yzAngle = (ui_descriptor.yzAngle == 270.f) ? 0.f : ui_descriptor.yzAngle + 90.f;
        ui_descriptor.sliceYZ->setAngle(ui_descriptor.yzAngle);
    }
}

void flipTexture(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    const WindowKind curWin = ui_descriptor.holovibes_.get_cd().current_window;

    if (curWin == WindowKind::XYview)
    {
        ui_descriptor.displayFlip = !ui_descriptor.displayFlip;
        ui_descriptor.mainDisplay->setFlip(ui_descriptor.displayFlip);
    }
    else if (ui_descriptor.sliceXZ && curWin == WindowKind::XZview)
    {
        ui_descriptor.xzFlip = !ui_descriptor.xzFlip;
        ui_descriptor.sliceXZ->setFlip(ui_descriptor.xzFlip);
    }
    else if (ui_descriptor.sliceYZ && curWin == WindowKind::YZview)
    {
        ui_descriptor.yzFlip = !ui_descriptor.yzFlip;
        ui_descriptor.sliceYZ->setFlip(ui_descriptor.yzFlip);
    }
}

bool set_contrast_mode(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    mainwindow.change_window();
    ui_descriptor.holovibes_.get_cd().contrast_enabled = value;
    ui_descriptor.holovibes_.get_cd().contrast_auto_refresh = true;
    pipe_refresh(ui_descriptor);
    return true;
}

void set_auto_contrast_cuts(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor.holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XZview);
        pipe->autocontrast_end_pipe(WindowKind::YZview);
    }
}

bool set_auto_contrast(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    try
    {
        if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor.holovibes_.get_compute_pipe().get()))
        {
            pipe->autocontrast_end_pipe(ui_descriptor.holovibes_.get_cd().current_window);
            return true;
        }
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what() << std::endl;
    }

    return false;
}

bool set_contrast_min(UserInterfaceDescriptor& ui_descriptor, const double value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    if (ui_descriptor.holovibes_.get_cd().contrast_enabled)
    {
        // FIXME: type issue, manipulatiion of double casted to float implies lost of data
        // Get the minimum contrast value rounded for the comparison
        const float old_val = ui_descriptor.holovibes_.get_cd().get_truncate_contrast_min(
            ui_descriptor.holovibes_.get_cd().current_window);
        // Floating number issue: cast to float for the comparison
        const float val = value;
        if (old_val != val)
        {
            ui_descriptor.holovibes_.get_cd().set_contrast_min(ui_descriptor.holovibes_.get_cd().current_window, value);
            pipe_refresh(ui_descriptor);
            return true;
        }
    }

    return false;
}

bool set_contrast_max(UserInterfaceDescriptor& ui_descriptor, const double value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    if (ui_descriptor.holovibes_.get_cd().contrast_enabled)
    {
        // FIXME: type issue, manipulatiion of double casted to float implies lost of data
        // Get the maximum contrast value rounded for the comparison
        const float old_val = ui_descriptor.holovibes_.get_cd().get_truncate_contrast_max(
            ui_descriptor.holovibes_.get_cd().current_window);
        // Floating number issue: cast to float for the comparison
        const float val = value;
        if (old_val != val)
        {
            ui_descriptor.holovibes_.get_cd().set_contrast_max(ui_descriptor.holovibes_.get_cd().current_window, value);
            pipe_refresh(ui_descriptor);
            return true;
        }
    }

    return false;
}

bool invert_contrast(UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    if (ui_descriptor.holovibes_.get_cd().contrast_enabled)
    {
        ui_descriptor.holovibes_.get_cd().contrast_invert = value;
        pipe_refresh(ui_descriptor);
        return true;
    }

    return false;
}

void set_auto_refresh_contrast(UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    ui_descriptor.holovibes_.get_cd().contrast_auto_refresh = value;
    pipe_refresh(ui_descriptor);
}

bool set_log_scale(UserInterfaceDescriptor& ui_descriptor, const bool value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    ui_descriptor.holovibes_.get_cd().set_log_scale_slice_enabled(ui_descriptor.holovibes_.get_cd().current_window,
                                                                  value);
    if (value && ui_descriptor.holovibes_.get_cd().contrast_enabled)
        set_auto_contrast(ui_descriptor);

    pipe_refresh(ui_descriptor);
    return true;
}

bool update_convo_kernel(UserInterfaceDescriptor& ui_descriptor, const std::string& value)
{
    LOG_INFO;

    if (ui_descriptor.holovibes_.get_cd().convolution_enabled)
    {
        ui_descriptor.holovibes_.get_cd().set_convolution(true, value);

        try
        {
            auto pipe = ui_descriptor.holovibes_.get_compute_pipe();
            pipe->request_convolution();
            // Wait for the convolution to be enabled for notify
            while (pipe->get_convolution_requested())
                continue;
        }
        catch (const std::exception& e)
        {
            ui_descriptor.holovibes_.get_cd().convolution_enabled = false;
            LOG_ERROR << e.what();
        }
        return true;
    }

    return false;
}

void set_divide_convolution_mode(UserInterfaceDescriptor& ui_descriptor, const bool value)
{
    LOG_INFO;

    ui_descriptor.holovibes_.get_cd().divide_convolution_enabled = value;

    pipe_refresh(ui_descriptor);
}

void display_reticle(UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    ui_descriptor.holovibes_.get_cd().reticle_enabled = value;
    if (value)
    {
        ui_descriptor.mainDisplay->getOverlayManager().create_overlay<::holovibes::gui::Reticle>();
        ui_descriptor.mainDisplay->getOverlayManager().create_default();
    }
    else
    {
        ui_descriptor.mainDisplay->getOverlayManager().disable_all(::holovibes::gui::Reticle);
    }

    pipe_refresh(ui_descriptor);
}

bool reticle_scale(UserInterfaceDescriptor& ui_descriptor, double value)
{
    LOG_INFO;

    if (0 > value || value > 1)
        return false;

    ui_descriptor.holovibes_.get_cd().reticle_scale = value;
    pipe_refresh(ui_descriptor);
    return true;
}

void record_finished(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    ui_descriptor.is_recording_ = false;
}

void activeNoiseZone(const UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    ui_descriptor.mainDisplay->getOverlayManager().create_overlay<::holovibes::gui::Noise>();
}

void activeSignalZone(const UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    ui_descriptor.mainDisplay->getOverlayManager().create_overlay<::holovibes::gui::Signal>();
}

void start_chart_display(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (ui_descriptor.holovibes_.get_cd().chart_display_enabled)
        return;

    auto pipe = ui_descriptor.holovibes_.get_compute_pipe();
    pipe->request_display_chart();

    // Wait for the chart display to be enabled for notify
    while (pipe->get_chart_display_requested())
        continue;

    ui_descriptor.plot_window_ = std::make_unique<::holovibes::gui::PlotWindow>(
        *ui_descriptor.holovibes_.get_compute_pipe()->get_chart_display_queue(),
        ui_descriptor.auto_scale_point_threshold_,
        "Chart");
}

void stop_chart_display(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (!ui_descriptor.holovibes_.get_cd().chart_display_enabled)
        return;

    try
    {
        auto pipe = ui_descriptor.holovibes_.get_compute_pipe();
        pipe->request_disable_display_chart();

        // Wait for the chart display to be disabled for notify
        while (pipe->get_disable_chart_display_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
    }

    ui_descriptor.plot_window_.reset(nullptr);
}

std::optional<bool>
update_lens_view(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    std::optional<bool> res = true;

    ui_descriptor.holovibes_.get_cd().gpu_lens_display_enabled = value;

    if (value)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos =
                ui_descriptor.mainDisplay->framePosition() + QPoint(ui_descriptor.mainDisplay->width() + 310, 0);
            ICompute* pipe = ui_descriptor.holovibes_.get_compute_pipe().get();

            const ::camera::FrameDescriptor& fd = ui_descriptor.holovibes_.get_gpu_input_queue()->get_fd();
            ushort lens_window_width = fd.width;
            ushort lens_window_height = fd.height;
            get_good_size(lens_window_width, lens_window_height, ui_descriptor.auxiliary_window_max_size);

            ui_descriptor.lens_window.reset(
                new ::holovibes::gui::RawWindow(pos,
                                                QSize(lens_window_width, lens_window_height),
                                                pipe->get_lens_queue().get(),
                                                ::holovibes::gui::KindOfView::Lens));

            ui_descriptor.lens_window->setTitle("Lens view");
            ui_descriptor.lens_window->setCd(&ui_descriptor.holovibes_.get_cd());
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what() << std::endl;
            res = std::nullopt;
        }
    }

    else
    {
        mainwindow.disable_lens_view();
        ui_descriptor.lens_window.reset(nullptr);
        res = false;
    }

    ::holovibes::api::pipe_refresh(ui_descriptor);
    return res;
}

void disable_lens_view(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    ui_descriptor.holovibes_.get_cd().gpu_lens_display_enabled = false;
    ui_descriptor.holovibes_.get_compute_pipe()->request_disable_lens_view();
}

std::optional<bool>
update_raw_view(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    std::optional<bool> res = true;

    if (value)
    {
        if (ui_descriptor.holovibes_.get_cd().batch_size > global::global_config.output_queue_max_size)
        {
            LOG_ERROR << "[RAW VIEW] Batch size must be lower than output queue size";
            return std::nullopt;
        }

        auto pipe = ui_descriptor.holovibes_.get_compute_pipe();
        pipe->request_raw_view();

        // Wait for the raw view to be enabled for notify
        while (pipe->get_raw_view_requested())
            continue;

        const ::camera::FrameDescriptor& fd = ui_descriptor.holovibes_.get_gpu_input_queue()->get_fd();
        ushort raw_window_width = fd.width;
        ushort raw_window_height = fd.height;
        get_good_size(raw_window_width, raw_window_height, ui_descriptor.auxiliary_window_max_size);

        // set positions of new windows according to the position of the main GL
        // window and Lens window
        QPoint pos = ui_descriptor.mainDisplay->framePosition() + QPoint(ui_descriptor.mainDisplay->width() + 310, 0);
        ui_descriptor.raw_window.reset(new ::holovibes::gui::RawWindow(pos,
                                                                       QSize(raw_window_width, raw_window_height),
                                                                       pipe->get_raw_view_queue().get()));

        ui_descriptor.raw_window->setTitle("Raw view");
        ui_descriptor.raw_window->setCd(&ui_descriptor.holovibes_.get_cd());
    }
    else
    {
        ui_descriptor.raw_window.reset(nullptr);
        mainwindow.disable_raw_view();
        res = false;
    }

    pipe_refresh(ui_descriptor);
    return res;
}

void disable_raw_view(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    auto pipe = ui_descriptor.holovibes_.get_compute_pipe();
    pipe->request_disable_raw_view();

    // Wait for the raw view to be disabled for notify
    while (pipe->get_disable_raw_view_requested())
        continue;
}

bool set_time_transformation_size(UserInterfaceDescriptor& ui_descriptor,
                                  int time_transformation_size,
                                  std::function<void()> callback)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    time_transformation_size = std::max(1, time_transformation_size);

    if (time_transformation_size == ui_descriptor.holovibes_.get_cd().time_transformation_size)
        return false;

    auto pipe = dynamic_cast<Pipe*>(ui_descriptor.holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect(callback);
    }

    return true;
}

void set_fft_shift(UserInterfaceDescriptor& ui_descriptor, const bool value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return;

    ui_descriptor.holovibes_.get_cd().fft_shift_enabled = value;
    pipe_refresh(ui_descriptor);
}

bool set_filter2d_n2(UserInterfaceDescriptor& ui_descriptor, int n)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    ui_descriptor.holovibes_.get_cd().filter2d_n2 = n;

    if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor.holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XYview);
        if (ui_descriptor.holovibes_.get_cd().time_transformation_cuts_enabled)
        {
            pipe->autocontrast_end_pipe(WindowKind::XZview);
            pipe->autocontrast_end_pipe(WindowKind::YZview);
        }
        if (ui_descriptor.holovibes_.get_cd().filter2d_view_enabled)
            pipe->autocontrast_end_pipe(WindowKind::Filter2D);
    }

    pipe_refresh(ui_descriptor);
    return true;
}

bool set_filter2d_n1(UserInterfaceDescriptor& ui_descriptor, int n)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    ui_descriptor.holovibes_.get_cd().filter2d_n1 = n;

    if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor.holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XYview);
        if (ui_descriptor.holovibes_.get_cd().time_transformation_cuts_enabled)
        {
            pipe->autocontrast_end_pipe(WindowKind::XZview);
            pipe->autocontrast_end_pipe(WindowKind::YZview);
        }
        if (ui_descriptor.holovibes_.get_cd().filter2d_view_enabled)
            pipe->autocontrast_end_pipe(WindowKind::Filter2D);
    }

    pipe_refresh(ui_descriptor);
    return true;
}

std::optional<bool>
update_filter2d_view(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, bool checked)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor))
        return std::nullopt;

    std::optional<bool> res = true;

    if (checked)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos =
                ui_descriptor.mainDisplay->framePosition() + QPoint(ui_descriptor.mainDisplay->width() + 310, 0);
            auto pipe = dynamic_cast<Pipe*>(ui_descriptor.holovibes_.get_compute_pipe().get());
            if (pipe)
            {
                pipe->request_filter2d_view();

                const camera::FrameDescriptor& fd = ui_descriptor.holovibes_.get_gpu_output_queue()->get_fd();
                ushort filter2d_window_width = fd.width;
                ushort filter2d_window_height = fd.height;
                get_good_size(filter2d_window_width, filter2d_window_height, ui_descriptor.auxiliary_window_max_size);

                // Wait for the filter2d view to be enabled for notify
                while (pipe->get_filter2d_view_requested())
                    continue;

                ui_descriptor.filter2d_window.reset(
                    new ::holovibes::gui::Filter2DWindow(pos,
                                                         QSize(filter2d_window_width, filter2d_window_height),
                                                         pipe->get_filter2d_view_queue().get(),
                                                         &mainwindow));

                ui_descriptor.filter2d_window->setTitle("Filter2D view");
                ui_descriptor.filter2d_window->setCd(&ui_descriptor.holovibes_.get_cd());

                ui_descriptor.holovibes_.get_cd().set_log_scale_slice_enabled(WindowKind::Filter2D, true);
                pipe->autocontrast_end_pipe(WindowKind::Filter2D);
            }
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what() << std::endl;
            res = false;
        }
    }

    else
    {
        mainwindow.disable_filter2d_view();
        ui_descriptor.filter2d_window.reset(nullptr);
        res = false;
    }

    pipe_refresh(ui_descriptor);
    return res;
}

void change_window(UserInterfaceDescriptor& ui_descriptor, const int index)
{
    LOG_INFO;

    if (index == 0)
        ui_descriptor.holovibes_.get_cd().current_window = WindowKind::XYview;
    else if (index == 1)
        ui_descriptor.holovibes_.get_cd().current_window = WindowKind::XZview;
    else if (index == 2)
        ui_descriptor.holovibes_.get_cd().current_window = WindowKind::YZview;
    else if (index == 3)
        ui_descriptor.holovibes_.get_cd().current_window = WindowKind::Filter2D;

    pipe_refresh(ui_descriptor);
}

void disable_filter2d_view(UserInterfaceDescriptor& ui_descriptor, const int index)
{
    LOG_INFO;

    auto pipe = ui_descriptor.holovibes_.get_compute_pipe();
    pipe->request_disable_filter2d_view();

    // Wait for the filter2d view to be disabled for notify
    while (pipe->get_disable_filter2d_view_requested())
        continue;

    // Change the focused window
    change_window(ui_descriptor, index);
}

bool set_filter2d(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, bool checked)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor))
        return false;

    if (checked)
    {
        if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor.holovibes_.get_compute_pipe().get()))
            pipe->autocontrast_end_pipe(WindowKind::XYview);
        ui_descriptor.holovibes_.get_cd().filter2d_enabled = checked;
    }
    else
    {
        ui_descriptor.holovibes_.get_cd().filter2d_enabled = checked;
        mainwindow.cancel_filter2d();
    }

    pipe_refresh(ui_descriptor);
    return true;
}

void toggle_renormalize(UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    ui_descriptor.holovibes_.get_cd().renorm_enabled = value;
    ui_descriptor.holovibes_.get_compute_pipe()->request_clear_img_acc();

    pipe_refresh(ui_descriptor);
}

} // namespace holovibes::api
