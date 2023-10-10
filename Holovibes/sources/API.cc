#include "API.hh"
#include "logger.hh"

namespace holovibes::api
{

#pragma region Local

void pipe_refresh()
{
    if (get_compute_mode() == Computation::Raw || UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    try
    {
        get_compute_pipe()->request_refresh();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR("{}", e.what());
    }
}
const QUrl get_documentation_url() { return QUrl("https://ftp.espci.fr/incoming/Atlan/holovibes/manual/"); }

const std::string get_credits()
{
    return "Holovibes v" + std::string(__HOLOVIBES_VERSION__) +
           "\n\n"

           "Developers:\n\n"

           "Adrien Langou\n"
           "Julien Nicolle\n"
           "Sacha Bellier\n"
           "David Chemaly\n"
           "Damien Didier\n"

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

#pragma region Close Compute

void camera_none()
{
    close_windows();
    close_critical_compute();

    if (get_compute_mode() == Computation::Hologram)

        Holovibes::instance().stop_compute();
    Holovibes::instance().stop_frame_read();

    UserInterfaceDescriptor::instance().is_enabled_camera_ = false;
    set_is_computation_stopped(true);

    UserInterfaceDescriptor::instance().import_type_ = ImportType::None;
}

#pragma endregion

#pragma region Cameras

bool change_camera(CameraKind c)
{
    LOG_FUNC(static_cast<int>(c));
    camera_none();

    if (c == CameraKind::NONE)
        return false;

    try
    {
        if (get_compute_mode() == Computation::Raw)
            Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();

        Holovibes::instance().start_camera_frame_read(c);
        UserInterfaceDescriptor::instance().is_enabled_camera_ = true;
        UserInterfaceDescriptor::instance().kCamera = c;

        set_is_computation_stopped(false);

        return true;
    }
    catch (const camera::CameraException& e)
    {
        LOG_ERROR("[CAMERA] {}", e.what());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }

    return false;
}

void configure_camera()
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(Holovibes::instance().get_camera_ini_name())));
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

void create_pipe()
{
    try
    {
        Holovibes::instance().start_compute();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR("cannot create Pipe: {}", e.what());
    }
}

void set_raw_mode(uint window_max_size)
{
    QPoint pos(0, 0);
    const camera::FrameDescriptor& fd = get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, window_max_size);
    QSize size(width, height);
    init_image_mode(pos, size);

    set_compute_mode(Computation::Raw);
    create_pipe(); // To remove ?

    UserInterfaceDescriptor::instance().mainDisplay.reset(
        new holovibes::gui::RawWindow(pos,
                                      size,
                                      get_gpu_input_queue().get(),
                                      static_cast<float>(width) / static_cast<float>(height)));
    UserInterfaceDescriptor::instance().mainDisplay->setTitle(QString("XY view"));
    UserInterfaceDescriptor::instance().mainDisplay->setBitshift(GSH::instance().get_raw_bitshift());
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
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(GSH::instance().get_rotation());
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(GSH::instance().get_flip_enabled());
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR("create_holo_window: {}", e.what());
    }
}

bool set_holographic_mode(ushort window_size)
{
    /* ---------- */
    try
    {
        set_compute_mode(Computation::Hologram);
        /* Pipe & Window */
        create_pipe();
        create_holo_window(window_size);
        /* Info Manager */
        auto fd = get_fd();
        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        /* Contrast */
        GSH::instance().set_contrast_enabled(true);

        return true;
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR("cannot set holographic mode: {}", e.what());
    }

    return false;
}

// TODO: param index is imposed by MainWindow behavior, and should be replaced by something more generic like
// dictionary
void refresh_view_mode(ushort window_size, uint index)
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

    set_img_type(static_cast<ImgType>(index));

    try
    {
        create_pipe();
        create_holo_window(window_size);
        UserInterfaceDescriptor::instance().mainDisplay->setScale(old_scale);
        UserInterfaceDescriptor::instance().mainDisplay->setTranslate(old_translation[0], old_translation[1]);
    }
    catch (const std::runtime_error& e)
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
        LOG_ERROR("refresh_view_mode: {}", e.what());
    }
}

void set_view_mode(const std::string& value, std::function<void()> callback)
{
    UserInterfaceDescriptor::instance().last_img_type_ = value;

    auto pipe = get_compute_pipe();

    pipe->insert_fn_end_vect(callback);
    pipe_refresh();

    // Force XYview autocontrast
    pipe->request_autocontrast(WindowKind::XYview);
    // Force cuts views autocontrast if needed
}

#pragma endregion

#pragma region Batch
// FIXME: Same function as under
void update_batch_size(std::function<void()> notify_callback, const uint batch_size)
{
    if (batch_size == api::get_batch_size())
        return;

    api::set_batch_size(batch_size);
    Holovibes::instance().get_compute_pipe()->request_update_batch_size();

    if (auto pipe = dynamic_cast<Pipe*>(get_compute_pipe().get()))
    {
        pipe->insert_fn_end_vect(notify_callback);
    }
    else
    {
        LOG_INFO("could not get pipe");
    }
}

#pragma endregion

#pragma region STFT

// FIXME: Same function as above
void update_time_stride(std::function<void()> callback, const uint time_stride)
{
    get_compute_pipe()->insert_fn_end_vect(callback);
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
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(GSH::instance().get_xz_rot());
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(GSH::instance().get_xz_flip_enabled());

        UserInterfaceDescriptor::instance().sliceYZ.reset(new gui::SliceWindow(
            yzPos,
            QSize(time_transformation_size, UserInterfaceDescriptor::instance().mainDisplay->height()),
            get_compute_pipe()->get_stft_slice_queue(1).get(),
            gui::KindOfView::SliceYZ));
        UserInterfaceDescriptor::instance().sliceYZ->setTitle("YZ view");
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(GSH::instance().get_yz_rot());
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(GSH::instance().get_yz_flip_enabled());

        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Cross>();
        GSH::instance().set_cuts_view_enabled(true);
        auto holo = dynamic_cast<gui::HoloWindow*>(UserInterfaceDescriptor::instance().mainDisplay.get());
        if (holo)
            holo->update_slice_transforms();

        pipe_refresh();

        return true;
    }
    catch (const std::logic_error& e)
    {
        LOG_ERROR("Catch {}", e.what());
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

    get_compute_pipe()->insert_fn_end_vect(callback);

    // Refresh pipe to remove cuts linked lambda from pipe
    pipe_refresh();
    GSH::instance().set_cuts_view_enabled(false);
}

#pragma endregion

#pragma region Computation

void change_window(const int index) { GSH::instance().change_window(index); }

void toggle_renormalize(bool value)
{
    set_renorm_enabled(value);

    if (UserInterfaceDescriptor::instance().import_type_ != ImportType::None)
        get_compute_pipe()->request_clear_img_acc();

    pipe_refresh();
}

void handle_update_exception()
{
    api::set_p_index(0);
    api::set_time_transformation_size(1);
    api::disable_convolution();
}

void set_filter2d(bool checked)
{
    set_filter2d_enabled(checked);
    set_auto_contrast_all();
}

void set_filter2d_view(bool checked, uint auxiliary_window_max_size)
{
    auto pipe = get_compute_pipe();
    if (checked)
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

        GSH::instance().set_log_scale_filter2d_enabled(true);
        pipe->request_autocontrast(WindowKind::Filter2D);
    }
    else
    {
        UserInterfaceDescriptor::instance().filter2d_window.reset(nullptr);
        pipe->request_disable_filter2d_view();
        while (pipe->get_disable_filter2d_view_requested())
            continue;
    }

    pipe_refresh();
}

void set_time_transformation_size(std::function<void()> callback) { get_compute_pipe()->insert_fn_end_vect(callback); }

void set_lens_view(bool checked, uint auxiliary_window_max_size)
{
    if (get_compute_mode() == Computation::Raw)
        return;

    set_lens_view_enabled(checked);

    auto pipe = get_compute_pipe();

    if (checked)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                         QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);

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
            LOG_ERROR("Catch {}", e.what());
        }
    }
    else
    {
        UserInterfaceDescriptor::instance().lens_window.reset(nullptr);

        pipe->request_disable_lens_view();
        while (pipe->get_disable_lens_view_requested())
            continue;

        pipe_refresh();
    }
}

void set_raw_view(bool checked, uint auxiliary_window_max_size)
{
    if (get_compute_mode() == Computation::Raw)
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

void set_p_accu_level(uint p_value)
{
    UserInterfaceDescriptor::instance().raw_window.reset(nullptr);

    GSH::instance().set_p_accu_level(p_value);
    pipe_refresh();
}

void set_x_accu_level(uint x_value)
{
    GSH::instance().set_x_accu_level(x_value);
    pipe_refresh();
}

void set_x_cuts(uint value)
{
    auto& holo = Holovibes::instance();
    const auto& fd = holo.get_gpu_input_queue()->get_fd();
    if (value < fd.width)
    {
        GSH::instance().set_x_cuts(value);
        pipe_refresh();
    }
}

void set_y_accu_level(uint y_value)
{
    GSH::instance().set_y_accu_level(y_value);
    pipe_refresh();
}

void set_y_cuts(uint value)
{
    auto& holo = Holovibes::instance();
    const auto& fd = holo.get_gpu_input_queue()->get_fd();
    if (value < fd.height)
    {
        GSH::instance().set_y_cuts(value);
        pipe_refresh();
    }
}

void set_x_y(uint x, uint y)
{

    GSH::instance().set_x_cuts(x);
    GSH::instance().set_y_cuts(y);
    pipe_refresh();
}

void set_q_index(uint value)
{
    GSH::instance().set_q_index(value);
    pipe_refresh();
}

void set_q_accu_level(uint value)
{
    GSH::instance().set_q_accu_level(value);
    pipe_refresh();
}
void set_p_index(uint value)
{
    GSH::instance().set_p_index(value);
    pipe_refresh();
}

void set_composite_intervals(int composite_p_red, int composite_p_blue)
{
    GSH::instance().set_rgb_p({composite_p_red, composite_p_blue});
    pipe_refresh();
}

void set_composite_intervals_hsv_h_min(uint composite_p_min_h)
{
    GSH::instance().set_composite_p_h({composite_p_min_h, GSH::instance().get_composite_p_max_h()});
    pipe_refresh();
}

void set_composite_intervals_hsv_h_max(uint composite_p_max_h)
{
    GSH::instance().set_composite_p_h({GSH::instance().get_composite_p_min_h(), composite_p_max_h});
    pipe_refresh();
}

void set_composite_intervals_hsv_s_min(uint composite_p_min_s)
{
    GSH::instance().set_composite_p_min_s(composite_p_min_s);
    pipe_refresh();
}

void set_composite_intervals_hsv_s_max(uint composite_p_max_s)
{
    GSH::instance().set_composite_p_max_s(composite_p_max_s);
    pipe_refresh();
}

void set_composite_intervals_hsv_v_min(uint composite_p_min_v)
{
    GSH::instance().set_composite_p_min_v(composite_p_min_v);
    pipe_refresh();
}

void set_composite_intervals_hsv_v_max(uint composite_p_max_v)
{
    GSH::instance().set_composite_p_max_v(composite_p_max_v);
    pipe_refresh();
}

void set_composite_weights(uint weight_r, uint weight_g, uint weight_b)
{
    GSH::instance().set_weight_rgb(weight_r, weight_g, weight_b);
    pipe_refresh();
}

void select_composite_rgb() { set_composite_kind(CompositeKind::RGB); }

void select_composite_hsv() { set_composite_kind(CompositeKind::HSV); }

void actualize_frequency_channel_s(bool composite_p_activated_s)
{
    GSH::instance().set_composite_p_activated_s(composite_p_activated_s);
}

void actualize_frequency_channel_v(bool composite_p_activated_v)
{
    GSH::instance().set_composite_p_activated_v(composite_p_activated_v);
}

void actualize_selection_h_gaussian_blur(bool h_blur_activated)
{
    GSH::instance().set_h_blur_activated(h_blur_activated);
}

void check_p_limits()
{
    int upper_bound = get_time_transformation_size() - 1;

    if (get_p_accu_level() > upper_bound)
        api::set_p_accu_level(upper_bound);

    upper_bound -= get_p_accu_level();

    if (upper_bound >= 0 && get_p_index() > static_cast<uint>(upper_bound))
        api::set_p_index(upper_bound);
}

void check_q_limits()
{
    int upper_bound = get_time_transformation_size() - 1;

    if (get_q_accu_level() > upper_bound)
        api::set_q_accu_level(upper_bound);

    upper_bound -= get_q_accu_level();

    if (upper_bound >= 0 && get_q_index() > static_cast<uint>(upper_bound))
        api::set_q_index(upper_bound);
}

void actualize_kernel_size_blur(uint h_blur_kernel_size) { GSH::instance().set_h_blur_kernel_size(h_blur_kernel_size); }

bool slide_update_threshold(
    const int slider_value, float& receiver, float& bound_to_update, const float lower_bound, const float upper_bound)
{
    receiver = slider_value / 1000.0f;

    if (lower_bound > upper_bound)
    {
        // FIXME bound_to_update = receiver ?
        bound_to_update = slider_value / 1000.0f;

        return true;
    }

    return false;
}

void set_wavelength(double value)
{
    set_lambda(static_cast<float>(value));

    pipe_refresh();
}

void set_z_distance(const double value)
{
    GSH::instance().set_z_distance(static_cast<float>(value));

    pipe_refresh();
}

void set_space_transformation(const SpaceTransformation value) { GSH::instance().set_space_transformation(value); }

void set_time_transformation(const TimeTransformation value) { GSH::instance().set_time_transformation(value); }

void set_unwrapping_2d(const bool value)
{
    get_compute_pipe()->request_unwrapping_2d(value);

    pipe_refresh();
}

void set_accumulation_level(int value)
{
    GSH::instance().set_accumulation_level(value);

    pipe_refresh();
}

void set_composite_area()
{
    UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::CompositeArea>();
}

void close_critical_compute()
{
    if (get_convolution_enabled())
        disable_convolution();

    if (api::get_cuts_view_enabled())
        cancel_time_transformation_cuts([]() {});

    Holovibes::instance().stop_compute();
}

void stop_all_worker_controller() { Holovibes::instance().stop_all_worker_controller(); }

unsigned get_img_accu_level() { return GSH::instance().get_img_accu_level(); }

int get_gpu_input_queue_fd_width() { return get_fd().width; }

int get_gpu_input_queue_fd_height() { return get_fd().height; }

float get_boundary() { return Holovibes::instance().get_boundary(); }

#pragma endregion

#pragma region Texture

static void change_angle()
{
    double rot = GSH::instance().get_rotation();
    double new_rot = (rot == 270.f) ? 0.f : rot + 90.f;

    GSH::instance().set_rotation(new_rot);
}

void rotateTexture()
{
    change_angle();

    if (GSH::instance().get_current_window_type() == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(GSH::instance().get_xy_rot());
    else if (UserInterfaceDescriptor::instance().sliceXZ &&
             GSH::instance().get_current_window_type() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(GSH::instance().get_xz_rot());
    else if (UserInterfaceDescriptor::instance().sliceYZ &&
             GSH::instance().get_current_window_type() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(GSH::instance().get_yz_rot());
}

static void change_flip() { GSH::instance().set_flip_enabled(!GSH::instance().get_flip_enabled()); }

void flipTexture()
{
    change_flip();

    if (GSH::instance().get_current_window_type() == WindowKind::XYview)
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(GSH::instance().get_xy_flip_enabled());
    else if (UserInterfaceDescriptor::instance().sliceXZ &&
             GSH::instance().get_current_window_type() == WindowKind::XZview)
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(GSH::instance().get_xz_flip_enabled());
    else if (UserInterfaceDescriptor::instance().sliceYZ &&
             GSH::instance().get_current_window_type() == WindowKind::YZview)
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(GSH::instance().get_yz_flip_enabled());
}

#pragma endregion

#pragma region Contrast - Log

void set_contrast_mode(bool value)
{
    GSH::instance().set_contrast_enabled(value);
    pipe_refresh();
}

void set_auto_contrast_cuts()
{
    auto pipe = get_compute_pipe();
    pipe->request_autocontrast(WindowKind::XZview);
    pipe->request_autocontrast(WindowKind::YZview);
}

bool set_auto_contrast()
{
    try
    {
        get_compute_pipe()->request_autocontrast(GSH::instance().get_current_window_type());
        return true;
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }

    return false;
}

void set_auto_contrast_all()
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    auto pipe = get_compute_pipe();
    pipe->request_autocontrast(WindowKind::XYview);
    if (api::get_cuts_view_enabled())
    {
        pipe->request_autocontrast(WindowKind::XZview);
        pipe->request_autocontrast(WindowKind::YZview);
    }
    if (get_filter2d_view_enabled())
        pipe->request_autocontrast(WindowKind::Filter2D);

    pipe_refresh();
}

void set_contrast_min(const double value)
{
    // Get the minimum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_min();
    // Floating number issue: cast to float for the comparison
    const float val = value;
    if (old_val != val)
    {
        GSH::instance().set_contrast_min(value);
        pipe_refresh();
    }
}

float get_truncate_contrast_max(const int precision)
{
    float value = GSH::instance().get_contrast_max();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

float get_truncate_contrast_min(const int precision)
{
    float value = GSH::instance().get_contrast_min();
    const double multiplier = std::pow(10.0, precision);
    return std::round(value * multiplier) / multiplier;
}

void set_contrast_max(const double value)
{
    // Get the maximum contrast value rounded for the comparison
    const float old_val = get_truncate_contrast_max();
    // Floating number issue: cast to float for the comparison
    const float val = value;
    if (old_val != val)
    {
        GSH::instance().set_contrast_max(value);
        pipe_refresh();
    }
}

void invert_contrast(bool value)
{
    GSH::instance().set_contrast_invert(value);
    pipe_refresh();
}

void set_auto_refresh_contrast(bool value)
{
    GSH::instance().set_contrast_auto_refresh(value);
    pipe_refresh();
}

void set_log_scale(const bool value)
{
    GSH::instance().set_log_scale_slice_enabled(value);
    if (value && GSH::instance().get_contrast_enabled())
        set_auto_contrast();

    pipe_refresh();
}

void set_raw_bitshift(unsigned int value) { GSH::instance().set_raw_bitshift(value); }

unsigned int get_raw_bitshift() { return GSH::instance().get_raw_bitshift(); }

float get_contrast_min() { return GSH::instance().get_contrast_min(); }

float get_contrast_max() { return GSH::instance().get_contrast_max(); }

bool get_contrast_invert_enabled() { return GSH::instance().get_contrast_invert(); }

bool get_img_log_scale_slice_enabled() { return GSH::instance().get_img_log_scale_slice_enabled(); }

#pragma endregion

#pragma region Convolution

void enable_convolution(const std::string& filename)
{
    GSH::instance().enable_convolution(filename == UID_CONVOLUTION_TYPE_DEFAULT ? std::nullopt
                                                                                : std::make_optional(filename));

    if (filename == UID_CONVOLUTION_TYPE_DEFAULT)
    {
        // Refresh because the current convolution might have change.
        pipe_refresh();
        return;
    }

    try
    {
        auto pipe = get_compute_pipe();
        pipe->request_convolution();
        // Wait for the convolution to be enabled for notify
        while (pipe->get_convolution_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        disable_convolution();
        LOG_ERROR("Catch {}", e.what());
    }
}

void disable_convolution()
{

    GSH::instance().disable_convolution();
    try
    {
        auto pipe = get_compute_pipe();
        pipe->request_disable_convolution();
        while (pipe->get_disable_convolution_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
    }
}

void set_divide_convolution(const bool value)
{
    if (value == get_divide_convolution_enabled())
        return;

    set_divide_convolution_enabled(value);
    pipe_refresh();
}

#pragma endregion

#pragma region Reticle

void display_reticle(bool value)
{
    set_reticle_display_enabled(value);

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
    set_reticle_scale(value);
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
        LOG_ERROR("Catch {}", e.what());
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
    LOG_FUNC(text);

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
        LOG_ERROR("Number of frames must be activated");
        return false;
    }

    if (batch_enabled && batch_input_path.empty())
    {
        LOG_ERROR("No batch input file");
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
    LOG_FUNC();

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
    LOG_FUNC();

    close_windows();
    close_critical_compute();

    Holovibes::instance().stop_all_worker_controller();
    Holovibes::instance().start_information_display();

    UserInterfaceDescriptor::instance().import_type_ = ImportType::None;

    set_is_computation_stopped(true);
}

bool import_start(size_t first_frame, size_t last_frame)
{
    LOG_FUNC(first_frame, last_frame);

    set_is_computation_stopped(false);

    // Because we are in file mode
    UserInterfaceDescriptor::instance().is_enabled_camera_ = false;

    try
    {

        Holovibes::instance().init_input_queue(UserInterfaceDescriptor::instance().file_fd_,
                                               api::get_input_buffer_size());
        // TODO remove
        Holovibes::instance().update_setting(settings::LoopOnInputFile{true});
        Holovibes::instance().start_file_frame_read(static_cast<unsigned int>(first_frame - 1),
                                                    static_cast<unsigned int>(last_frame - first_frame + 1));
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
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
