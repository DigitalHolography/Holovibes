#include "API.hh"

namespace holovibes::api
{

#pragma region Local

void pipe_refresh()
{
    if (is_raw_mode() || UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    try
    {
        get_compute_pipe()->request_refresh();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what();
    }
}

const QUrl get_documentation_url() { return QUrl("https://ftp.espci.fr/incoming/Atlan/holovibes/manual/"); }

const std::string get_credits()
{
    return "Holovibes v" + std::string(__HOLOVIBES_VERSION__) +
           "\n\n"

           "Developers:\n\n"

           "Philippe Bernet\n"
           "Eliott Bouhana\n"
           "Fabien Colmagro\n"
           "Marius Dubosc\n"
           "Guillaume Poisson\n"

           "Anthony Strazzella\n"
           "Ilan Guenet\n"
           "Nicolas Blin\n"
           "Quentin Kaci\n"
           "Theo Lepage\n"

           "Loïc Bellonnet-Mottet\n"
           "Antoine Martin\n"
           "François Te\n"

           "Ellena Davoine\n"
           "Clement Fang\n"
           "Danae Marmai\n"
           "Hugo Verjus\n"

           "Eloi Charpentier\n"
           "Julien Gautier\n"
           "Florian Lapeyre\n"

           "Thomas Jarrossay\n"
           "Alexandre Bartz\n"

           "Cyril Cetre\n"
           "Clement Ledant\n"

           "Eric Delanghe\n"
           "Arnaud Gaillard\n"
           "Geoffrey Le Gourrierec\n"

           "Jeffrey Bencteux\n"
           "Thomas Kostas\n"
           "Pierre Pagnoux\n"

           "Antoine Dillée\n"
           "Romain Cancillière\n"

           "Michael Atlan\n";
}

bool is_raw_mode() { return get_cd().get_compute_mode() == Computation::Raw; }

bool is_gpu_input_queue() { return get_gpu_input_queue() != nullptr; }

void close_windows()
{
    UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);

    UserInterfaceDescriptor::instance().sliceXZ.reset(nullptr);
    UserInterfaceDescriptor::instance().sliceYZ.reset(nullptr);
    UserInterfaceDescriptor::instance().filter2d_window.reset(nullptr);

    if (UserInterfaceDescriptor::instance().lens_window)
        set_lens_view(false, 0);
    if (UserInterfaceDescriptor::instance().raw_window)
        set_raw_view(false, 0);

    UserInterfaceDescriptor::instance().plot_window_.reset(nullptr);
}

#pragma endregion

#pragma region Ini

void save_user_preferences(boost::property_tree::ptree& ptree)
{
    // Display
    ptree.put<ushort>("display.rate", get_display_rate());
    // Step
    ptree.put<uint>("gui_settings.record_frame_step", UserInterfaceDescriptor::instance().record_frame_step_);
    // Camera
    ptree.put<int>("image_rendering.camera", static_cast<int>(UserInterfaceDescriptor::instance().kCamera));
    // Chart
    ptree.put<size_t>("chart.auto_scale_point_threshold",
                      UserInterfaceDescriptor::instance().auto_scale_point_threshold_);
    // Files
    ptree.put<std::string>("files.default_output_filename",
                           UserInterfaceDescriptor::instance().default_output_filename_);
    ptree.put<std::string>("files.record_output_directory",
                           UserInterfaceDescriptor::instance().record_output_directory_);
    ptree.put<std::string>("files.file_input_directory", UserInterfaceDescriptor::instance().file_input_directory_);
    ptree.put<std::string>("files.batch_input_directory", UserInterfaceDescriptor::instance().batch_input_directory_);
}
void load_user_preferences(const boost::property_tree::ptree& ptree)
{
    // Display
    set_display_rate(ptree.get<uint>("display.rate", get_display_rate()));
    // Step
    UserInterfaceDescriptor::instance().record_frame_step_ =
        ptree.get<uint>("gui_settings.record_frame_step_", UserInterfaceDescriptor::instance().record_frame_step_);
    // Chart
    UserInterfaceDescriptor::instance().auto_scale_point_threshold_ =
        ptree.get<size_t>("chart.auto_scale_point_threshold",
                          UserInterfaceDescriptor::instance().auto_scale_point_threshold_);
    // Camera
    UserInterfaceDescriptor::instance().kCamera = static_cast<CameraKind>(
        ptree.get<int>("image_rendering.camera", static_cast<int>(UserInterfaceDescriptor::instance().kCamera)));
    // Files
    UserInterfaceDescriptor::instance().default_output_filename_ =
        ptree.get<std::string>("files.default_output_filename",
                               UserInterfaceDescriptor::instance().default_output_filename_);
    UserInterfaceDescriptor::instance().record_output_directory_ =
        ptree.get<std::string>("files.record_output_directory",
                               UserInterfaceDescriptor::instance().record_output_directory_);
    UserInterfaceDescriptor::instance().file_input_directory_ =
        ptree.get<std::string>("files.file_input_directory", UserInterfaceDescriptor::instance().file_input_directory_);
    UserInterfaceDescriptor::instance().batch_input_directory_ =
        ptree.get<std::string>("files.batch_input_directory",
                               UserInterfaceDescriptor::instance().batch_input_directory_);
}

#pragma endregion

#pragma region Close Compute

void camera_none()
{
    close_windows();
    close_critical_compute();

    if (!is_raw_mode())
        Holovibes::instance().stop_compute();
    Holovibes::instance().stop_frame_read();

    UserInterfaceDescriptor::instance().is_enabled_camera_ = false;
    get_cd().set_computation_stopped(true);

    UserInterfaceDescriptor::instance().import_type_ = ImportType::None;
}

#pragma endregion

#pragma region Cameras

bool change_camera(CameraKind c)
{
    camera_none();

    if (c == CameraKind::NONE)
        return false;

    try
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
        if (!is_raw_mode())
            Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();

        Holovibes::instance().start_camera_frame_read(c);
        UserInterfaceDescriptor::instance().is_enabled_camera_ = true;
        UserInterfaceDescriptor::instance().kCamera = c;

        get_cd().set_computation_stopped(false);

        return true;
    }
    catch (const camera::CameraException& e)
    {
        LOG_ERROR << "[CAMERA] " << e.what();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
    }

    return false;
}

void configure_camera()
{
    auto path = std::filesystem::path(settings::camera_config_folderpath) / Holovibes::instance().get_camera_ini_name();
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(path.string())));
}

#pragma endregion

#pragma region Image Mode

void init_image_mode(QPoint& position, QSize& size)
{
    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        position = UserInterfaceDescriptor::instance().mainDisplay->framePosition();
        size = UserInterfaceDescriptor::instance().mainDisplay->size();
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
    }
}

void create_pipe(Observer& observer)
{
    try
    {
        Holovibes::instance().start_compute();
        get_compute_pipe()->register_observer(observer);
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot create Pipe: " << e.what();
    }
}

void func_to_rename_display_start(Observer& observer, ushort window_max_size)
{
    close_windows();

    bool save_convo = get_convolution_enabled();
    close_critical_compute();

    if (get_compute_mode() == Computation::Raw)
        set_raw_mode(observer, window_max_size);
    else
    {
        set_holographic_mode(observer, window_max_size);
        // TODO: Add all settings that need the pipe to be added: Contrast...

        set_convolution_mode(save_convo);
    }
}

void set_raw_mode(Observer& observer, uint window_max_size)
{
    QPoint pos(0, 0);
    const camera::FrameDescriptor& fd = get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, window_max_size);
    QSize size(width, height);
    init_image_mode(pos, size);
    get_cd().set_compute_mode(Computation::Raw);
    UserInterfaceDescriptor::instance().mainDisplay.reset(
        new holovibes::gui::RawWindow(pos,
                                      size,
                                      get_gpu_input_queue().get(),
                                      static_cast<float>(width) / static_cast<float>(height)));
    UserInterfaceDescriptor::instance().mainDisplay->setTitle(QString("XY view"));
    std::string fd_info =
        std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
}

void create_holo_window(ushort window_size)
{
    QPoint pos(0, 0);
    const camera::FrameDescriptor& fd = get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, window_size);
    QSize size(width, height);
    init_image_mode(pos, size);

    try
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(
            new gui::HoloWindow(pos,
                                size,
                                get_gpu_output_queue().get(),
                                get_compute_pipe(),
                                UserInterfaceDescriptor::instance().sliceXZ,
                                UserInterfaceDescriptor::instance().sliceYZ,
                                static_cast<float>(width) / static_cast<float>(height)));
        UserInterfaceDescriptor::instance().mainDisplay->set_is_resize(false);
        UserInterfaceDescriptor::instance().mainDisplay->setTitle(QString("XY view"));
        UserInterfaceDescriptor::instance().mainDisplay->resetTransform();
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(get_cd().get_rotation());
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(get_cd().get_flip_enabled());
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "create_holo_window: " << e.what();
    }
}

bool set_holographic_mode(Observer& observer, ushort window_size)
{
    /* ---------- */
    try
    {
        get_cd().set_compute_mode(Computation::Hologram);
        /* Pipe & Window */
        create_pipe(observer);
        create_holo_window(window_size);
        /* Info Manager */
        auto fd = get_fd();
        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        /* Contrast */
        get_cd().set_contrast_enabled(true);

        return true;
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot set holographic mode: " << e.what();
    }

    return false;
}

// TODO: param index is imposed by MainWindow behavior, and should be replaced by something more generic like
// dictionary
void refresh_view_mode(Observer& observer, ushort window_size, uint index)
{
    float old_scale = 1.f;
    glm::vec2 old_translation(0.f, 0.f);
    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        old_scale = UserInterfaceDescriptor::instance().mainDisplay->getScale();
        old_translation = UserInterfaceDescriptor::instance().mainDisplay->getTranslate();
    }

    close_windows();
    close_critical_compute();

    get_cd().set_img_type(static_cast<ImgType>(index));

    try
    {
        create_pipe(observer);
        create_holo_window(window_size);
        UserInterfaceDescriptor::instance().mainDisplay->setScale(old_scale);
        UserInterfaceDescriptor::instance().mainDisplay->setTranslate(old_translation[0], old_translation[1]);
    }
    catch (const std::runtime_error& e)
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
        LOG_ERROR << "refresh_view_mode: " << e.what();
    }
}

void set_view_mode(const std::string& value, std::function<void()> callback)
{
    UserInterfaceDescriptor::instance().last_img_type_ = value;

    auto pipe = dynamic_cast<Pipe*>(get_compute_pipe().get());

    pipe->insert_fn_end_vect(callback);
    pipe_refresh();

    // Force XYview autocontrast
    pipe->autocontrast_end_pipe(WindowKind::XYview);
    // Force cuts views autocontrast if needed
}

#pragma endregion

#pragma region Batch
// FIXME: Same fucntion as under
void update_batch_size(std::function<void()> callback, const uint batch_size)
{
    if (auto pipe = dynamic_cast<Pipe*>(get_compute_pipe().get()))
        pipe->insert_fn_end_vect(callback);
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

#pragma endregion

#pragma region STFT

// FIXME: Same fucntion as above
void update_time_transformation_stride(std::function<void()> callback, const uint time_transformation_stride)
{
    if (auto pipe = dynamic_cast<Pipe*>(get_compute_pipe().get()))
        pipe->insert_fn_end_vect(callback);
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

bool set_3d_cuts_view(uint time_transformation_size)
{
    try
    {
        get_compute_pipe()->create_stft_slice_queue();
        // set positions of new windows according to the position of the
        // main GL window
        QPoint xzPos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                       QPoint(0, UserInterfaceDescriptor::instance().mainDisplay->height() + 42);
        QPoint yzPos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                       QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 20, 0);

        while (get_compute_pipe()->get_update_time_transformation_size_request())
            continue;
        while (get_compute_pipe()->get_cuts_request())
            continue;

        UserInterfaceDescriptor::instance().sliceXZ.reset(new gui::SliceWindow(
            xzPos,
            QSize(UserInterfaceDescriptor::instance().mainDisplay->width(), time_transformation_size),
            get_compute_pipe()->get_stft_slice_queue(0).get(),
            gui::KindOfView::SliceXZ));
        UserInterfaceDescriptor::instance().sliceXZ->setTitle("XZ view");
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(get_cd().get_xz_rot());
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(get_cd().get_xz_flip_enabled());

        UserInterfaceDescriptor::instance().sliceYZ.reset(new gui::SliceWindow(
            yzPos,
            QSize(time_transformation_size, UserInterfaceDescriptor::instance().mainDisplay->height()),
            get_compute_pipe()->get_stft_slice_queue(1).get(),
            gui::KindOfView::SliceYZ));
        UserInterfaceDescriptor::instance().sliceYZ->setTitle("YZ view");
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(get_cd().get_yz_rot());
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(get_cd().get_yz_flip_enabled());

        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Cross>();
        set_3d_cuts_view_enabled(true);
        auto holo = dynamic_cast<gui::HoloWindow*>(UserInterfaceDescriptor::instance().mainDisplay.get());
        if (holo)
            holo->update_slice_transforms();
        return true;
    }
    catch (const std::logic_error& e)
    {
        LOG_ERROR << e.what() << std::endl;
    }

    return false;
}

void cancel_time_transformation_cuts(std::function<void()> callback)
{
    UserInterfaceDescriptor::instance().sliceXZ.reset(nullptr);
    UserInterfaceDescriptor::instance().sliceYZ.reset(nullptr);

    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        UserInterfaceDescriptor::instance().mainDisplay->setCursor(Qt::ArrowCursor);
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::SliceCross);
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::Cross);
    }

    get_compute_pipe().get()->insert_fn_end_vect(callback);

    try
    {
        // Wait for refresh to be enabled for notify
        while (get_compute_pipe()->get_refresh_request())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
    }

    set_3d_cuts_view_enabled(false);
}

#pragma endregion

#pragma region Computation

void change_window(const int index) { get_cd().change_window(index); }

void toggle_renormalize(bool value)
{
    get_cd().set_renorm_enabled(value);

    if (UserInterfaceDescriptor::instance().import_type_ != ImportType::None)
        get_compute_pipe()->request_clear_img_acc();

    pipe_refresh();
}

void set_filter2d(bool checked)
{
    get_cd().set_filter2d_enabled(checked);
    set_auto_contrast_all();
}

void set_filter2d_view(bool checked, uint auxiliary_window_max_size)
{
    if (checked)
    {
        if (auto pipe = get_compute_pipe())
        {
            pipe->request_filter2d_view();
            while (pipe->get_filter2d_view_requested())
                continue;

            const camera::FrameDescriptor& fd = get_fd();
            ushort filter2d_window_width = fd.width;
            ushort filter2d_window_height = fd.height;
            get_good_size(filter2d_window_width, filter2d_window_height, auxiliary_window_max_size);

            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                         QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);
            UserInterfaceDescriptor::instance().filter2d_window.reset(
                new gui::Filter2DWindow(pos,
                                        QSize(filter2d_window_width, filter2d_window_height),
                                        pipe->get_filter2d_view_queue().get()));

            UserInterfaceDescriptor::instance().filter2d_window->setTitle("Filter2D view");

            get_cd().set_log_scale_filter2d_enabled(true);
            pipe->autocontrast_end_pipe(WindowKind::Filter2D);
        }
    }
    else
    {
        UserInterfaceDescriptor::instance().filter2d_window.reset(nullptr);

        auto pipe = get_compute_pipe();
        pipe->request_disable_filter2d_view();
        while (pipe->get_disable_filter2d_view_requested())
            continue;
    }

    pipe_refresh();
}

void set_fft_shift(const bool value)
{
    get_cd().set_fft_shift_enabled(value);

    pipe_refresh();
}

void set_time_transformation_size(std::function<void()> callback)
{
    auto pipe = dynamic_cast<Pipe*>(get_compute_pipe().get());
    if (pipe)
        pipe->insert_fn_end_vect(callback);
}

void set_lens_view(bool checked, uint auxiliary_window_max_size)
{
    if (is_raw_mode())
        return;

    get_cd().set_lens_view_enabled(checked);

    if (checked)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                         QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);
            ICompute* pipe = get_compute_pipe().get();

            const ::camera::FrameDescriptor& fd = get_fd();
            ushort lens_window_width = fd.width;
            ushort lens_window_height = fd.height;
            get_good_size(lens_window_width, lens_window_height, auxiliary_window_max_size);

            UserInterfaceDescriptor::instance().lens_window.reset(
                new gui::RawWindow(pos,
                                   QSize(lens_window_width, lens_window_height),
                                   pipe->get_lens_queue().get(),
                                   0.f,
                                   gui::KindOfView::Lens));

            UserInterfaceDescriptor::instance().lens_window->setTitle("Lens view");
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what() << std::endl;
        }
    }
    else
    {
        UserInterfaceDescriptor::instance().lens_window.reset(nullptr);

        auto pipe = get_compute_pipe();
        pipe->request_disable_lens_view();
        while (pipe->get_disable_lens_view_requested())
            continue;

        pipe_refresh();
    }
}

void set_raw_view(bool checked, uint auxiliary_window_max_size)
{
    if (is_raw_mode())
        return;

    auto pipe = get_compute_pipe();

    if (checked)
    {
        pipe->request_raw_view();
        while (pipe->get_raw_view_requested())
            continue;

        const ::camera::FrameDescriptor& fd = get_fd();
        ushort raw_window_width = fd.width;
        ushort raw_window_height = fd.height;
        get_good_size(raw_window_width, raw_window_height, auxiliary_window_max_size);

        // set positions of new windows according to the position of the main GL
        // window and Lens window
        QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                     QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);
        UserInterfaceDescriptor::instance().raw_window.reset(
            new gui::RawWindow(pos, QSize(raw_window_width, raw_window_height), pipe->get_raw_view_queue().get()));

        UserInterfaceDescriptor::instance().raw_window->setTitle("Raw view");
    }
    else
    {
        UserInterfaceDescriptor::instance().raw_window.reset(nullptr);

        pipe->request_disable_raw_view();
        while (pipe->get_disable_raw_view_requested())
            continue;
    }

    pipe_refresh();
}

void set_p_accu(uint p_value)
{
    UserInterfaceDescriptor::instance().raw_window.reset(nullptr);

    get_cd().set_p_accu_level(p_value);
    pipe_refresh();
}

void set_x_accu(uint x_value)
{
    get_cd().set_x_accu_level(x_value);
    pipe_refresh();
}

void set_y_accu(uint y_value)
{
    get_cd().set_y_accu_level(y_value);
    pipe_refresh();
}

void set_x_y(uint x, uint y)
{
    /* TODO: app logic as to be in ManWindow
            // frame_descriptor can be unvalid
            const camera::FrameDescriptor& frame_descriptor = get_fd();

            if (x < frame_descriptor.width)
                get_cd().set_x_cuts(x);

            if (y < frame_descriptor.height)
                get_cd().set_y_cuts(y);
    */

    get_cd().set_x_cuts(x);
    get_cd().set_y_cuts(y);
}

void set_q(int value) { get_cd().set_q_index(value); }

void set_q_accu(uint q_value)
{
    get_cd().set_q_accu_level(q_value);
    pipe_refresh();
}

void set_p(int value)
{
    get_cd().set_p_index(value);

    pipe_refresh();
}

void set_composite_intervals(uint composite_p_red, uint composite_p_blue)
{
    get_cd().set_rgb_p_min(composite_p_red);
    get_cd().set_rgb_p_max(composite_p_blue);

    pipe_refresh();
}

void set_composite_intervals_hsv_h_min(uint composite_p_min_h)
{
    get_cd().set_composite_p_min_h(composite_p_min_h);
    pipe_refresh();
}

void set_composite_intervals_hsv_h_max(uint composite_p_max_h)
{
    get_cd().set_composite_p_max_h(composite_p_max_h);
    pipe_refresh();
}

void set_composite_intervals_hsv_s_min(uint composite_p_min_s)
{
    get_cd().set_composite_p_min_s(composite_p_min_s);
    pipe_refresh();
}

void set_composite_intervals_hsv_s_max(uint composite_p_max_s)
{
    get_cd().set_composite_p_max_s(composite_p_max_s);
    pipe_refresh();
}

void set_composite_intervals_hsv_v_min(uint composite_p_min_v)
{
    get_cd().set_composite_p_min_v(composite_p_min_v);
    pipe_refresh();
}

void set_composite_intervals_hsv_v_max(uint composite_p_max_v)
{
    get_cd().set_composite_p_max_v(composite_p_max_v);
    pipe_refresh();
}

void set_composite_weights(uint weight_r, uint weight_g, uint weight_b)
{
    get_cd().set_weight_rgb(weight_r, weight_g, weight_b);
    pipe_refresh();
}

void set_composite_auto_weights(bool value) { get_cd().set_composite_auto_weights(value); }

void set_composite_kind(const CompositeKind& value) { get_cd().set_composite_kind(value); }
void select_composite_rgb() { get_cd().set_composite_kind(CompositeKind::RGB); }

void select_composite_hsv() { get_cd().set_composite_kind(CompositeKind::HSV); }

void actualize_frequency_channel_s(bool composite_p_activated_s)
{
    get_cd().set_composite_p_activated_s(composite_p_activated_s);
}

void actualize_frequency_channel_v(bool composite_p_activated_v)
{
    get_cd().set_composite_p_activated_v(composite_p_activated_v);
}

void actualize_selection_h_gaussian_blur(bool h_blur_activated) { get_cd().set_h_blur_activated(h_blur_activated); }

void actualize_kernel_size_blur(uint h_blur_kernel_size) { get_cd().set_h_blur_kernel_size(h_blur_kernel_size); }

bool slide_update_threshold(
    const int slider_value, float& receiver, float& bound_to_update, const float lower_bound, const float upper_bound)
{
    // Store the slider value in ui_descriptor_.holovibes_.get_cd() (ComputeDescriptor)
    receiver = slider_value / 1000.0f;

    if (lower_bound > upper_bound)
    {
        // FIXME bound_to_update = receiver ?
        bound_to_update = slider_value / 1000.0f;

        return true;
    }

    return false;
}

void set_wavelength(const double value)
{
    get_cd().set_lambda(static_cast<float>(value));

    pipe_refresh();
}

void set_z_distance(const double value)
{
    get_cd().set_zdistance(static_cast<float>(value));

    pipe_refresh();
}

void set_space_transformation(const SpaceTransformation& value) { get_cd().set_space_transformation(value); }

void set_time_transformation(const TimeTransformation& value) { get_cd().set_time_transformation(value); }

void adapt_time_transformation_stride_to_batch_size() { get_cd().adapt_time_transformation_stride(); }

void set_unwrapping_2d(const bool value)
{
    get_compute_pipe()->request_unwrapping_2d(value);

    pipe_refresh();
}

void set_accumulation_level(int value)
{
    get_cd().set_accumulation_level(value);

    pipe_refresh();
}

void set_composite_area()
{
    UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::CompositeArea>();
}

void close_critical_compute()
{
    if (get_cd().convolution_enabled)
        set_convolution_mode(false);

    if (get_cd().time_transformation_cuts_enabled)
        cancel_time_transformation_cuts([]() { return; });

    Holovibes::instance().stop_compute();
}

void stop_all_worker_controller() { Holovibes::instance().stop_all_worker_controller(); }

unsigned get_img_accu_level() { return get_cd().get_img_accu_level(); }

int get_gpu_input_queue_fd_width() { return get_fd().width; }

int get_gpu_input_queue_fd_height() { return get_fd().height; }

float get_boundary() { return Holovibes::instance().get_boundary(); }

#pragma endregion

#pragma region Texture

void rotateTexture()
{
    get_cd().change_angle();

    if (get_cd().current_window == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(get_cd().get_xy_rot());
    else if (UserInterfaceDescriptor::instance().sliceXZ && get_cd().get_current_window() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(get_cd().get_xz_rot());
    else if (UserInterfaceDescriptor::instance().sliceYZ && get_cd().get_current_window() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(get_cd().get_yz_rot());
}

void flipTexture()
{
    get_cd().change_flip();

    if (get_cd().get_current_window() == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(get_cd().get_xy_flip_enabled());
    else if (UserInterfaceDescriptor::instance().sliceXZ && get_cd().get_current_window() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(get_cd().get_xz_flip_enabled());
    else if (UserInterfaceDescriptor::instance().sliceYZ && get_cd().get_current_window() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(get_cd().get_yz_flip_enabled());
}

#pragma endregion

#pragma region Contrast - Log

void set_contrast_mode(bool value)
{
    get_cd().set_contrast_enabled(value);
    pipe_refresh();
}

void set_auto_contrast_cuts()
{
    if (auto pipe = dynamic_cast<Pipe*>(get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XZview);
        pipe->autocontrast_end_pipe(WindowKind::YZview);
    }
}

bool set_auto_contrast()
{
    try
    {
        if (auto pipe = dynamic_cast<Pipe*>(get_compute_pipe().get()))
        {
            pipe->autocontrast_end_pipe(get_cd().current_window);
            return true;
        }
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what() << std::endl;
    }

    return false;
}

void set_auto_contrast_all()
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    if (auto pipe = dynamic_cast<Pipe*>(get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XYview);
        if (get_cd().time_transformation_cuts_enabled)
        {
            pipe->autocontrast_end_pipe(WindowKind::XZview);
            pipe->autocontrast_end_pipe(WindowKind::YZview);
        }
        if (get_filter2d_view_enabled())
            pipe->autocontrast_end_pipe(WindowKind::Filter2D);

        pipe_refresh();
    }
}

void set_contrast_min(const double value)
{
    // Get the minimum contrast value rounded for the comparison
    const float old_val = get_cd().get_truncate_contrast_min();
    // Floating number issue: cast to float for the comparison
    const float val = value;
    if (old_val != val)
    {
        get_cd().set_contrast_min(value);
        pipe_refresh();
    }
}

void set_contrast_max(const double value)
{
    // Get the maximum contrast value rounded for the comparison
    const float old_val = get_cd().get_truncate_contrast_max();
    // Floating number issue: cast to float for the comparison
    const float val = value;
    if (old_val != val)
    {
        get_cd().set_contrast_max(value);
        pipe_refresh();
    }
}

void invert_contrast(bool value)
{
    get_cd().set_contrast_invert(value);
    pipe_refresh();
}

void set_auto_refresh_contrast(bool value)
{
    get_cd().set_contrast_auto_refresh(value);
    pipe_refresh();
}

void set_log_scale(const bool value)
{
    get_cd().set_log_scale_slice_enabled(value);
    if (value && get_cd().get_contrast_enabled())
        set_auto_contrast();

    pipe_refresh();
}

float get_contrast_min() { return get_cd().get_contrast_min(); }

float get_contrast_max() { return get_cd().get_contrast_max(); }

bool get_contrast_invert_enabled() { return get_cd().get_contrast_invert(); }

bool get_img_log_scale_slice_enabled() { return get_cd().get_img_log_scale_slice_enabled(); }

void check_batch_size_limit() { get_cd().check_batch_size_limit(); }

#pragma endregion

#pragma region Convolution

void update_convo_kernel(const std::string& value)
{
    if (UserInterfaceDescriptor::instance().import_type_ == None)
        return;

    get_cd().set_convolution(true, value);
    UserInterfaceDescriptor::instance().convo_name = value;

    try
    {
        auto pipe = get_compute_pipe();
        pipe->request_convolution();
        while (pipe->get_convolution_requested())
            continue;
    }
    catch (const std::exception&)
    {
        get_cd().set_convolution_enabled(false);
    }
}

void set_convolution_mode(bool value)
{
    get_cd().set_convolution_enabled(value);

    if (value)
    {
        if (UserInterfaceDescriptor::instance().convo_name != UID_CONVOLUTION_TYPE_DEFAULT)
            update_convo_kernel(UserInterfaceDescriptor::instance().convo_name);
    }
    else
    {
        try
        {
            auto pipe = get_compute_pipe();
            pipe->request_disable_convolution();
            while (pipe->get_disable_convolution_requested())
                continue;
        }
        catch (const std::exception&)
        {
            get_cd().set_convolution_enabled(false);
        }
    }
}

void set_divide_convolution(const bool value)
{
    get_cd().set_divide_convolution_enabled(value);
    pipe_refresh();
}

#pragma endregion

#pragma region Reticle

void display_reticle(bool value)
{
    if (value == get_cd().get_reticle_display_enabled())
        return;

    get_cd().set_reticle_display_enabled(value);
    if (value)
    {
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Reticle>();
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_default();
    }
    else
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::Reticle);

    pipe_refresh();
}

void reticle_scale(float value)
{
    get_cd().set_reticle_scale(value);
    pipe_refresh();
}

#pragma endregion

#pragma region Chart

void active_noise_zone()
{
    UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Noise>();
}

void active_signal_zone()
{
    UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Signal>();
}

void start_chart_display()
{
    auto pipe = get_compute_pipe();
    pipe->request_display_chart();

    // Wait for the chart display to be enabled for notify
    while (pipe->get_chart_display_requested())
        continue;

    UserInterfaceDescriptor::instance().plot_window_ =
        std::make_unique<gui::PlotWindow>(*get_compute_pipe()->get_chart_display_queue(),
                                          UserInterfaceDescriptor::instance().auto_scale_point_threshold_,
                                          "Chart");
}

void stop_chart_display()
{
    try
    {
        auto pipe = get_compute_pipe();
        pipe->request_disable_display_chart();

        // Wait for the chart display to be disabled for notify
        while (pipe->get_disable_chart_display_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
    }

    UserInterfaceDescriptor::instance().plot_window_.reset(nullptr);
}

#pragma endregion

#pragma region Record

const std::string browse_record_output_file(std::string& std_filepath)
{
    // FIXME: path separator should depend from system
    std::replace(std_filepath.begin(), std_filepath.end(), '/', '\\');
    std::filesystem::path path = std::filesystem::path(std_filepath);

    // FIXME Opti: we could be all these 3 operations below on a single string processing
    UserInterfaceDescriptor::instance().record_output_directory_ = path.parent_path().string();
    const std::string file_ext = path.extension().string();
    UserInterfaceDescriptor::instance().default_output_filename_ = path.stem().string();

    return file_ext;
}

void set_record_mode(const std::string& text)
{
    // TODO: Dictionnary
    if (text == "Chart")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::CHART;
    else if (text == "Processed Image")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::HOLOGRAM;
    else if (text == "Raw Image")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::RAW;
    else if (text == "3D Cuts XZ")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::CUTS_XZ;
    else if (text == "3D Cuts YZ")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::CUTS_YZ;
    else
        throw std::exception("Record mode not handled");
}

bool start_record_preconditions(const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const std::string& batch_input_path)
{
    // Preconditions to start record

    if (!nb_frame_checked)
        nb_frames_to_record = std::nullopt;

    if ((batch_enabled || UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART) &&
        nb_frames_to_record == std::nullopt)
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

void start_record(const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback)
{
    if (batch_enabled)
    {
        Holovibes::instance().start_batch_gpib(batch_input_path,
                                               output_path,
                                               nb_frames_to_record.value(),
                                               UserInterfaceDescriptor::instance().record_mode_,
                                               callback);
    }
    else
    {
        if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
        {
            Holovibes::instance().start_chart_record(output_path, nb_frames_to_record.value(), callback);
        }
        else
        {
            Holovibes::instance().start_frame_record(output_path,
                                                     nb_frames_to_record,
                                                     UserInterfaceDescriptor::instance().record_mode_,
                                                     0,
                                                     callback);
        }
    }
}

void stop_record()
{
    Holovibes::instance().stop_batch_gpib();

    if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
        Holovibes::instance().stop_chart_record();
    else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::HOLOGRAM ||
             UserInterfaceDescriptor::instance().record_mode_ == RecordMode::RAW ||
             UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CUTS_XZ ||
             UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CUTS_YZ)
        Holovibes::instance().stop_frame_record();
}

void record_finished() { UserInterfaceDescriptor::instance().is_recording_ = false; }

#pragma endregion

#pragma region Import

void import_stop()
{
    Holovibes::instance().stop_all_worker_controller();
    Holovibes::instance().start_information_display();

    close_critical_compute();

    UserInterfaceDescriptor::instance().import_type_ = ImportType::None;

    get_cd().set_computation_stopped(true);
}

bool import_start(
    std::string& file_path, unsigned int fps, size_t first_frame, bool load_file_in_gpu, size_t last_frame)
{
    get_cd().set_computation_stopped(false);

    bool res = true;

    // Because we are in file mode
    UserInterfaceDescriptor::instance().is_enabled_camera_ = false;

    try
    {

        Holovibes::instance().init_input_queue(UserInterfaceDescriptor::instance().file_fd_,
                                               get_cd().get_input_buffer_size());
        Holovibes::instance().start_file_frame_read(file_path,
                                                    true,
                                                    fps,
                                                    static_cast<unsigned int>(first_frame - 1),
                                                    static_cast<unsigned int>(last_frame - first_frame + 1),
                                                    load_file_in_gpu);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        UserInterfaceDescriptor::instance().is_enabled_camera_ = false;
        Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();
        return false;
    }

    UserInterfaceDescriptor::instance().is_enabled_camera_ = true;
    UserInterfaceDescriptor::instance().import_type_ = ImportType::File;

    return true;
}

std::optional<io_files::InputFrameFile*> import_file(const std::string& filename)
{
    if (!filename.empty())
    {

        // Will throw if the file format (extension) cannot be handled
        auto input_file = io_files::InputFrameFileFactory::open(filename);

        return input_file;
    }

    return std::nullopt;
}

#pragma endregion

#pragma region Advanced Settings
void open_advanced_settings(QMainWindow* parent, ::holovibes::gui::AdvancedSettingsWindowPanel* specific_panel)
{
    UserInterfaceDescriptor::instance().is_advanced_settings_displayed = true;
    UserInterfaceDescriptor::instance().advanced_settings_window_ =
        std::make_unique<::holovibes::gui::AdvancedSettingsWindow>(parent, specific_panel);
}

#pragma endregion

#pragma region Information

void start_information_display(const std::function<void()>& callback)
{
    Holovibes::instance().start_information_display(callback);
}

#pragma endregion

std::unique_ptr<::holovibes::gui::RawWindow>& get_main_display()
{
    return UserInterfaceDescriptor::instance().mainDisplay;
}

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_xz() { return UserInterfaceDescriptor::instance().sliceXZ; }

std::unique_ptr<::holovibes::gui::SliceWindow>& get_slice_yz() { return UserInterfaceDescriptor::instance().sliceYZ; }

std::unique_ptr<::holovibes::gui::RawWindow>& get_lens_window()
{
    return UserInterfaceDescriptor::instance().lens_window;
}

std::unique_ptr<::holovibes::gui::RawWindow>& get_raw_window()
{
    return UserInterfaceDescriptor::instance().raw_window;
}

std::unique_ptr<::holovibes::gui::Filter2DWindow>& get_filter2d_window()
{
    return UserInterfaceDescriptor::instance().filter2d_window;
}

} // namespace holovibes::api
