#include "api.hh"

#include <optional>
namespace holovibes::api
{

#pragma region Local

// VALID
void open_file(const std::string& path)
{
    LOG_INFO;
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString(path.c_str())));
}

void pipe_refresh()
{
    LOG_INFO;
    if (is_raw_mode())
        return;

    try
    {
        if (!Holovibes::instance().get_compute_pipe()->get_request_refresh())
            Holovibes::instance().get_compute_pipe()->request_refresh();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what();
    }
}

// VALID
bool init_holovibes_import_mode(
    std::string& file_path, unsigned int fps, size_t first_frame, bool load_file_in_gpu, size_t last_frame)
{
    LOG_INFO;

    // Set the image rendering ui params
    Holovibes::instance().get_cd().time_transformation_stride = std::ceil(static_cast<float>(fps) / 20.0f);
    Holovibes::instance().get_cd().batch_size = 1;

    // Because we are in import mode
    UserInterfaceDescriptor::instance().is_enabled_camera_ = false;

    try
    {

        Holovibes::instance().init_input_queue(UserInterfaceDescriptor::instance().file_fd_);
        Holovibes::instance().start_file_frame_read(file_path,
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
        UserInterfaceDescriptor::instance().is_enabled_camera_ = false;
        Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();
        return false;
    }
    UserInterfaceDescriptor::instance().is_enabled_camera_ = true;
    return true;
}

// VALID
const QUrl get_documentation_url() { return QUrl("https://ftp.espci.fr/incoming/Atlan/holovibes/manual/"); }

// VALID
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

// VALID
bool is_raw_mode()
{
    LOG_INFO;
    return Holovibes::instance().get_cd().compute_mode == Computation::Raw;
}

// VALID
bool is_gpu_input_queue()
{
    LOG_INFO;
    return Holovibes::instance().get_gpu_input_queue() != nullptr;
}

// VALID
void remove_infos()
{
    LOG_INFO;
    Holovibes::instance().get_info_container().clear();
}

// VALID
void close_windows()
{
    LOG_INFO;
    UserInterfaceDescriptor::instance().sliceXZ.reset(nullptr);
    UserInterfaceDescriptor::instance().sliceYZ.reset(nullptr);

    UserInterfaceDescriptor::instance().plot_window_.reset(nullptr);
    UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);

    UserInterfaceDescriptor::instance().lens_window.reset(nullptr);
    Holovibes::instance().get_cd().gpu_lens_display_enabled = false;

    UserInterfaceDescriptor::instance().filter2d_window.reset(nullptr);
    Holovibes::instance().get_cd().filter2d_view_enabled = false;

    /* Raw view & recording */
    UserInterfaceDescriptor::instance().raw_window.reset(nullptr);
    Holovibes::instance().get_cd().raw_view_enabled = false;

    // Disable overlays
    Holovibes::instance().get_cd().reticle_enabled = false;
}

#pragma endregion

#pragma region Ini

// VALID
void configure_holovibes()
{
    LOG_INFO;
    open_file(ini::get_global_ini_path());
}

// VALID
void load_ini(const std::string& path, boost::property_tree::ptree& ptree)
{
    LOG_INFO;

    boost::property_tree::ini_parser::read_ini(path, ptree);

    if (!ptree.empty())
    {
        // Load general compute data
        ini::load_ini(ptree, Holovibes::instance().get_cd());

        // Load window specific data
        UserInterfaceDescriptor::instance().default_output_filename_ =
            ptree.get<std::string>("files.default_output_filename",
                                   UserInterfaceDescriptor::instance().default_output_filename_);
        UserInterfaceDescriptor::instance().record_output_directory_ =
            ptree.get<std::string>("files.record_output_directory",
                                   UserInterfaceDescriptor::instance().record_output_directory_);
        UserInterfaceDescriptor::instance().file_input_directory_ =
            ptree.get<std::string>("files.file_input_directory",
                                   UserInterfaceDescriptor::instance().file_input_directory_);
        UserInterfaceDescriptor::instance().batch_input_directory_ =
            ptree.get<std::string>("files.batch_input_directory",
                                   UserInterfaceDescriptor::instance().batch_input_directory_);
        /*
                const float z_step = ptree.get<float>("image_rendering.z_step",
           UserInterfaceDescriptor::instance().z_step_); if (z_step > 0.0f) mainwindow.set_z_step(z_step);
        */
        UserInterfaceDescriptor::instance().last_img_type_ =
            Holovibes::instance().get_cd().img_type == ImgType::Composite
                ? "Composite image"
                : UserInterfaceDescriptor::instance().last_img_type_;

        UserInterfaceDescriptor::instance().displayAngle =
            ptree.get("view.mainWindow_rotate", UserInterfaceDescriptor::instance().displayAngle);
        UserInterfaceDescriptor::instance().xzAngle =
            ptree.get<float>("view.xCut_rotate", UserInterfaceDescriptor::instance().xzAngle);
        UserInterfaceDescriptor::instance().yzAngle =
            ptree.get<float>("view.yCut_rotate", UserInterfaceDescriptor::instance().yzAngle);
        UserInterfaceDescriptor::instance().displayFlip =
            ptree.get("view.mainWindow_flip", UserInterfaceDescriptor::instance().displayFlip);
        UserInterfaceDescriptor::instance().xzFlip =
            ptree.get("view.xCut_flip", UserInterfaceDescriptor::instance().xzFlip);
        UserInterfaceDescriptor::instance().yzFlip =
            ptree.get("view.yCut_flip", UserInterfaceDescriptor::instance().yzFlip);

        UserInterfaceDescriptor::instance().auto_scale_point_threshold_ =
            ptree.get<size_t>("chart.auto_scale_point_threshold",
                              UserInterfaceDescriptor::instance().auto_scale_point_threshold_);
        /*
                const uint record_frame_step =
                    ptree.get<uint>("record.record_frame_step", UserInterfaceDescriptor::instance().record_frame_step_);
                mainwindow.set_record_frame_step(record_frame_step);
        */
        UserInterfaceDescriptor::instance().window_max_size = ptree.get<uint>("display.main_window_max_size", 768);
        UserInterfaceDescriptor::instance().time_transformation_cuts_window_max_size =
            ptree.get<uint>("display.time_transformation_cuts_window_max_size", 512);
        UserInterfaceDescriptor::instance().auxiliary_window_max_size =
            ptree.get<uint>("display.auxiliary_window_max_size", 512);
    }
}

// VALID
void save_ini(const std::string& path, boost::property_tree::ptree& ptree)
{
    LOG_INFO;

    // Save general compute data
    ini::save_ini(ptree, Holovibes::instance().get_cd());

    // Save window specific data
    ptree.put<std::string>("files.default_output_filename",
                           UserInterfaceDescriptor::instance().default_output_filename_);
    ptree.put<std::string>("files.record_output_directory",
                           UserInterfaceDescriptor::instance().record_output_directory_);
    ptree.put<std::string>("files.file_input_directory", UserInterfaceDescriptor::instance().file_input_directory_);
    ptree.put<std::string>("files.batch_input_directory", UserInterfaceDescriptor::instance().batch_input_directory_);

    ptree.put<int>("image_rendering.camera", static_cast<int>(UserInterfaceDescriptor::instance().kCamera));

    ptree.put<double>("image_rendering.z_step", UserInterfaceDescriptor::instance().z_step_);

    ptree.put<float>("view.mainWindow_rotate", UserInterfaceDescriptor::instance().displayAngle);
    ptree.put<float>("view.xCut_rotate", UserInterfaceDescriptor::instance().xzAngle);
    ptree.put<float>("view.yCut_rotate", UserInterfaceDescriptor::instance().yzAngle);
    ptree.put<int>("view.mainWindow_flip", UserInterfaceDescriptor::instance().displayFlip);
    ptree.put<int>("view.xCut_flip", UserInterfaceDescriptor::instance().xzFlip);
    ptree.put<int>("view.yCut_flip", UserInterfaceDescriptor::instance().yzFlip);

    ptree.put<size_t>("chart.auto_scale_point_threshold",
                      UserInterfaceDescriptor::instance().auto_scale_point_threshold_);

    ptree.put<uint>("record.record_frame_step", UserInterfaceDescriptor::instance().record_frame_step_);

    ptree.put<uint>("display.main_window_max_size", UserInterfaceDescriptor::instance().window_max_size);
    ptree.put<uint>("display.time_transformation_cuts_window_max_size",
                    UserInterfaceDescriptor::instance().time_transformation_cuts_window_max_size);
    ptree.put<uint>("display.auxiliary_window_max_size", UserInterfaceDescriptor::instance().auxiliary_window_max_size);

    boost::property_tree::write_ini(path, ptree);
}

#pragma endregion

#pragma region Close Compute

// VALID
void camera_none()
{
    LOG_INFO;
    close_windows();
    close_critical_compute();
    if (!is_raw_mode())
        Holovibes::instance().stop_compute();
    Holovibes::instance().stop_frame_read();
    remove_infos();

    UserInterfaceDescriptor::instance().is_enabled_camera_ = false;
    Holovibes::instance().get_cd().is_computation_stopped = true;
}

#pragma endregion

#pragma region Cameras

// VALID
bool change_camera(CameraKind c, const Computation computation)
{
    LOG_INFO;

    bool res = false;

    try
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
        if (!is_raw_mode())
            Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();

        set_camera_timeout();

        set_computation_mode(computation);

        Holovibes::instance().start_camera_frame_read(c);
        UserInterfaceDescriptor::instance().is_enabled_camera_ = true;
        UserInterfaceDescriptor::instance().import_type_ = UserInterfaceDescriptor::ImportType::Camera;
        UserInterfaceDescriptor::instance().kCamera = c;

        Holovibes::instance().get_cd().is_computation_stopped = false;

        res = true;
    }
    catch (const camera::CameraException& e)
    {
        LOG_ERROR << "[CAMERA] " << e.what();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
    }

    return res;
}

// VALID
void configure_camera()
{
    LOG_INFO;
    open_file(std::filesystem::current_path().generic_string() + "/" + Holovibes::instance().get_camera_ini_path());
}

// VALID
void set_camera_timeout()
{
    LOG_INFO;
    camera::FRAME_TIMEOUT = global::global_config.frame_timeout;
}

#pragma endregion

#pragma region Image Mode

// VALID
void init_image_mode(QPoint& position, QSize& size)
{
    LOG_INFO;

    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        position = UserInterfaceDescriptor::instance().mainDisplay->framePosition();
        size = UserInterfaceDescriptor::instance().mainDisplay->size();
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
    }
}

// TODO: change parameter type to Observer
void set_raw_mode(gui::MainWindow& observer)
{
    LOG_INFO;

    QPoint pos(0, 0);
    const camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, UserInterfaceDescriptor::instance().window_max_size);
    QSize size(width, height);
    init_image_mode(pos, size);
    Holovibes::instance().get_cd().compute_mode = Computation::Raw;
    createPipe(observer);
    UserInterfaceDescriptor::instance().mainDisplay.reset(
        new holovibes::gui::RawWindow(pos, size, Holovibes::instance().get_gpu_input_queue().get()));
    UserInterfaceDescriptor::instance().mainDisplay->setTitle(QString("XY view"));
    UserInterfaceDescriptor::instance().mainDisplay->setCd(&Holovibes::instance().get_cd());
    UserInterfaceDescriptor::instance().mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));
    std::string fd_info =
        std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
    Holovibes::instance().get_info_container().add_indication(InformationContainer::IndicationType::INPUT_FORMAT,
                                                              fd_info);
    unset_convolution_mode();
    set_divide_convolution_mode(false);
}

// TODO: change parameter type to Observer
void createPipe(gui::MainWindow& observer)
{
    LOG_INFO;
    try
    {
        Holovibes::instance().start_compute();
        Holovibes::instance().get_compute_pipe()->register_observer(observer);
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot create Pipe: " << e.what();
    }
}

// TODO: change parameter type to Observer
void createHoloWindow(gui::MainWindow& observer)
{
    LOG_INFO;
    QPoint pos(0, 0);
    const camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, UserInterfaceDescriptor::instance().window_max_size);
    QSize size(width, height);
    init_image_mode(pos, size);
    /* ---------- */
    try
    {
        // TODO: ctor should take Observer as last param instead of MainWindow
        UserInterfaceDescriptor::instance().mainDisplay.reset(
            new gui::HoloWindow(pos,
                                size,
                                Holovibes::instance().get_gpu_output_queue().get(),
                                Holovibes::instance().get_compute_pipe(),
                                UserInterfaceDescriptor::instance().sliceXZ,
                                UserInterfaceDescriptor::instance().sliceYZ,
                                &observer));
        UserInterfaceDescriptor::instance().mainDisplay->set_is_resize(false);
        UserInterfaceDescriptor::instance().mainDisplay->setTitle(QString("XY view"));
        UserInterfaceDescriptor::instance().mainDisplay->setCd(&Holovibes::instance().get_cd());
        UserInterfaceDescriptor::instance().mainDisplay->resetTransform();
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(UserInterfaceDescriptor::instance().displayAngle);
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(UserInterfaceDescriptor::instance().displayFlip);
        UserInterfaceDescriptor::instance().mainDisplay->setRatio(static_cast<float>(width) /
                                                                  static_cast<float>(height));
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "createHoloWindow: " << e.what();
    }
}

// TODO: change parameter type to Observer
bool set_holographic_mode(gui::MainWindow& observer, camera::FrameDescriptor& fd)
{
    LOG_INFO;

    /* ---------- */
    try
    {
        Holovibes::instance().get_cd().compute_mode = Computation::Hologram;
        /* Pipe & Window */
        createPipe(observer);
        createHoloWindow(observer);
        /* Info Manager */
        fd = Holovibes::instance().get_gpu_output_queue()->get_fd();
        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        Holovibes::instance().get_info_container().add_indication(InformationContainer::IndicationType::OUTPUT_FORMAT,
                                                                  fd_info);
        /* Contrast */
        Holovibes::instance().get_cd().contrast_enabled = true;

        return true;
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot set holographic mode: " << e.what();
    }

    return false;
}

// TODO: change parameter type to Observer
void refreshViewMode(gui::MainWindow& observer, uint index)
{
    LOG_INFO;
    float old_scale = 1.f;
    glm::vec2 old_translation(0.f, 0.f);
    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        old_scale = UserInterfaceDescriptor::instance().mainDisplay->getScale();
        old_translation = UserInterfaceDescriptor::instance().mainDisplay->getTranslate();
    }

    close_windows();
    close_critical_compute();

    Holovibes::instance().get_cd().img_type = static_cast<ImgType>(index);

    try
    {
        createPipe(observer);
        createHoloWindow(observer);
        UserInterfaceDescriptor::instance().mainDisplay->setScale(old_scale);
        UserInterfaceDescriptor::instance().mainDisplay->setTranslate(old_translation[0], old_translation[1]);
    }
    catch (const std::runtime_error& e)
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
        LOG_ERROR << "refreshViewMode: " << e.what();
    }
}

// VALID
void set_view_mode(const std::string& value, std::function<void()> callback)
{
    LOG_INFO;

    UserInterfaceDescriptor::instance().last_img_type_ = value;

    auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get());

    pipe->insert_fn_end_vect(callback);
    pipe_refresh();

    // Force XYview autocontrast
    pipe->autocontrast_end_pipe(WindowKind::XYview);
    // Force cuts views autocontrast if needed
}

#pragma endregion

#pragma region Batch

// VALID
void update_batch_size(std::function<void()> callback, const uint batch_size)
{
    LOG_INFO;

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
        pipe->insert_fn_end_vect(callback);
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

#pragma endregion

#pragma region STFT

// VALID
void update_time_transformation_stride(std::function<void()> callback, const uint time_transformation_stride)
{
    LOG_INFO;

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
        pipe->insert_fn_end_vect(callback);
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

// TODO: change parameter type to Observer
bool toggle_time_transformation_cuts(gui::MainWindow& observer)
{
    LOG_INFO;

    // if checked
    try
    {
        Holovibes::instance().get_compute_pipe()->create_stft_slice_queue();
        // set positions of new windows according to the position of the
        // main GL window
        QPoint xzPos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                       QPoint(0, UserInterfaceDescriptor::instance().mainDisplay->height() + 42);
        QPoint yzPos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                       QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 20, 0);
        const ushort nImg = Holovibes::instance().get_cd().time_transformation_size;
        uint time_transformation_size = std::max(256u, std::min(512u, (uint)nImg));

        if (time_transformation_size > UserInterfaceDescriptor::instance().time_transformation_cuts_window_max_size)
            time_transformation_size = UserInterfaceDescriptor::instance().time_transformation_cuts_window_max_size;

        while (Holovibes::instance().get_compute_pipe()->get_update_time_transformation_size_request())
            continue;
        while (Holovibes::instance().get_compute_pipe()->get_cuts_request())
            continue;

        UserInterfaceDescriptor::instance().sliceXZ.reset(new gui::SliceWindow(
            xzPos,
            QSize(UserInterfaceDescriptor::instance().mainDisplay->width(), time_transformation_size),
            Holovibes::instance().get_compute_pipe()->get_stft_slice_queue(0).get(),
            gui::KindOfView::SliceXZ,
            &observer));
        UserInterfaceDescriptor::instance().sliceXZ->setTitle("XZ view");
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(UserInterfaceDescriptor::instance().xzAngle);
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(UserInterfaceDescriptor::instance().xzFlip);
        UserInterfaceDescriptor::instance().sliceXZ->setCd(&Holovibes::instance().get_cd());

        UserInterfaceDescriptor::instance().sliceYZ.reset(new gui::SliceWindow(
            yzPos,
            QSize(time_transformation_size, UserInterfaceDescriptor::instance().mainDisplay->height()),
            Holovibes::instance().get_compute_pipe()->get_stft_slice_queue(1).get(),
            gui::KindOfView::SliceYZ,
            &observer));
        UserInterfaceDescriptor::instance().sliceYZ->setTitle("YZ view");
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(UserInterfaceDescriptor::instance().yzAngle);
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(UserInterfaceDescriptor::instance().yzFlip);
        UserInterfaceDescriptor::instance().sliceYZ->setCd(&Holovibes::instance().get_cd());

        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Cross>();
        Holovibes::instance().get_cd().time_transformation_cuts_enabled = true;
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

// VALID
void cancel_time_transformation_cuts(std::function<void()> callback)
{
    LOG_INFO;

    Holovibes::instance().get_cd().contrast_max_slice_xz = false;
    Holovibes::instance().get_cd().contrast_max_slice_yz = false;
    Holovibes::instance().get_cd().log_scale_slice_xz_enabled = false;
    Holovibes::instance().get_cd().log_scale_slice_yz_enabled = false;
    Holovibes::instance().get_cd().img_acc_slice_xz_enabled = false;
    Holovibes::instance().get_cd().img_acc_slice_yz_enabled = false;

    Holovibes::instance().get_compute_pipe().get()->insert_fn_end_vect(callback);

    try
    {
        // Wait for refresh to be enabled for notify
        while (Holovibes::instance().get_compute_pipe()->get_refresh_request())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
    }

    Holovibes::instance().get_cd().time_transformation_cuts_enabled = false;

    UserInterfaceDescriptor::instance().sliceXZ.reset(nullptr);
    UserInterfaceDescriptor::instance().sliceYZ.reset(nullptr);

    if (UserInterfaceDescriptor::instance().mainDisplay)
    {
        UserInterfaceDescriptor::instance().mainDisplay->setCursor(Qt::ArrowCursor);
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::SliceCross);
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::Cross);
    }
}

#pragma endregion

#pragma region Computation

// VALID
void change_window(const int index)
{
    LOG_INFO;

    if (index == 0)
        Holovibes::instance().get_cd().current_window = WindowKind::XYview;
    else if (index == 1)
        Holovibes::instance().get_cd().current_window = WindowKind::XZview;
    else if (index == 2)
        Holovibes::instance().get_cd().current_window = WindowKind::YZview;
    else if (index == 3)
        Holovibes::instance().get_cd().current_window = WindowKind::Filter2D;

    pipe_refresh();
}

// VALID
void toggle_renormalize(bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().renorm_enabled = value;
    Holovibes::instance().get_compute_pipe()->request_clear_img_acc();

    pipe_refresh();
}

// VALID
void set_filter2d()
{
    LOG_INFO;

    Holovibes::instance().get_cd().filter2d_enabled = true;

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
        pipe->autocontrast_end_pipe(WindowKind::XYview);

    pipe_refresh();
}

void disable_filter2d_view(const int index)
{
    LOG_INFO;

    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_disable_filter2d_view();

    // Wait for the filter2d view to be disabled for notify
    while (pipe->get_disable_filter2d_view_requested())
        continue;

    // Change the focused window
    change_window(index);
}

std::optional<bool> update_filter2d_view(gui::MainWindow& mainwindow, bool checked)
{
    LOG_INFO;
    if (api::is_raw_mode())
        return std::nullopt;

    std::optional<bool> res = true;

    if (checked)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                         QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);
            auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get());
            if (pipe)
            {
                pipe->request_filter2d_view();

                const camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_output_queue()->get_fd();
                ushort filter2d_window_width = fd.width;
                ushort filter2d_window_height = fd.height;
                get_good_size(filter2d_window_width,
                              filter2d_window_height,
                              UserInterfaceDescriptor::instance().auxiliary_window_max_size);

                // Wait for the filter2d view to be enabled for notify
                while (pipe->get_filter2d_view_requested())
                    continue;

                UserInterfaceDescriptor::instance().filter2d_window.reset(
                    new gui::Filter2DWindow(pos,
                                            QSize(filter2d_window_width, filter2d_window_height),
                                            pipe->get_filter2d_view_queue().get(),
                                            &mainwindow));

                UserInterfaceDescriptor::instance().filter2d_window->setTitle("Filter2D view");
                UserInterfaceDescriptor::instance().filter2d_window->setCd(&Holovibes::instance().get_cd());

                Holovibes::instance().get_cd().set_log_scale_slice_enabled(WindowKind::Filter2D, true);
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
        UserInterfaceDescriptor::instance().filter2d_window.reset(nullptr);
        res = false;
    }

    pipe_refresh();
    return res;
}

// VALID
void cancel_filter2d()
{
    LOG_INFO;

    Holovibes::instance().get_cd().filter2d_enabled = false;
}

// VALID
void set_fft_shift(const bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().fft_shift_enabled = value;
    pipe_refresh();
}

// VALID
void set_time_transformation_size(std::function<void()> callback)
{
    LOG_INFO;

    auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get());
    if (pipe)
        pipe->insert_fn_end_vect(callback);
}

// VALID
bool set_lens_view()
{
    LOG_INFO;

    bool res = false;

    try
    {
        // set positions of new windows according to the position of the
        // main GL window
        QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                     QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);
        ICompute* pipe = Holovibes::instance().get_compute_pipe().get();

        const ::camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();
        ushort lens_window_width = fd.width;
        ushort lens_window_height = fd.height;
        get_good_size(lens_window_width,
                      lens_window_height,
                      UserInterfaceDescriptor::instance().auxiliary_window_max_size);

        UserInterfaceDescriptor::instance().lens_window.reset(
            new gui::RawWindow(pos,
                               QSize(lens_window_width, lens_window_height),
                               pipe->get_lens_queue().get(),
                               gui::KindOfView::Lens));

        UserInterfaceDescriptor::instance().lens_window->setTitle("Lens view");
        UserInterfaceDescriptor::instance().lens_window->setCd(&Holovibes::instance().get_cd());
        res = true;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what() << std::endl;
    }

    api::pipe_refresh();
    return res;
}

// VALID
void disable_lens_view()
{
    LOG_INFO;

    Holovibes::instance().get_cd().gpu_lens_display_enabled = false;
    Holovibes::instance().get_compute_pipe()->request_disable_lens_view();
    UserInterfaceDescriptor::instance().lens_window.reset(nullptr);
    api::pipe_refresh();
}

// VALID
void set_raw_view()
{
    LOG_INFO;

    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_raw_view();

    // Wait for the raw view to be enabled for notify
    while (pipe->get_raw_view_requested())
        continue;

    const ::camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();
    ushort raw_window_width = fd.width;
    ushort raw_window_height = fd.height;
    get_good_size(raw_window_width, raw_window_height, UserInterfaceDescriptor::instance().auxiliary_window_max_size);

    // set positions of new windows according to the position of the main GL
    // window and Lens window
    QPoint pos = UserInterfaceDescriptor::instance().mainDisplay->framePosition() +
                 QPoint(UserInterfaceDescriptor::instance().mainDisplay->width() + 310, 0);
    UserInterfaceDescriptor::instance().raw_window.reset(
        new gui::RawWindow(pos, QSize(raw_window_width, raw_window_height), pipe->get_raw_view_queue().get()));

    UserInterfaceDescriptor::instance().raw_window->setTitle("Raw view");
    UserInterfaceDescriptor::instance().raw_window->setCd(&Holovibes::instance().get_cd());

    pipe_refresh();
}

// VALID
void disable_raw_view()
{
    LOG_INFO;

    UserInterfaceDescriptor::instance().raw_window.reset(nullptr);

    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_disable_raw_view();

    // Wait for the raw view to be disabled for notify
    while (pipe->get_disable_raw_view_requested())
        continue;

    pipe_refresh();
}

// VALID
void set_p_accu(bool is_p_accu, uint p_value)
{
    UserInterfaceDescriptor::instance().raw_window.reset(nullptr);
    Holovibes::instance().get_cd().p_accu_enabled = is_p_accu;
    Holovibes::instance().get_cd().p_acc_level = p_value;
    pipe_refresh();
}

// VALID
void set_x_accu(bool is_x_accu, uint x_value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().x_accu_enabled = is_x_accu;
    Holovibes::instance().get_cd().x_acc_level = x_value;
    pipe_refresh();
}

// VALID
void set_y_accu(bool is_y_accu, uint y_value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().y_accu_enabled = is_y_accu;
    Holovibes::instance().get_cd().y_acc_level = y_value;
    pipe_refresh();
}

// VALID
void set_x_y(uint x, uint y)
{
    LOG_INFO;

    const camera::FrameDescriptor& frame_descriptor = get_fd();

    if (x < frame_descriptor.width)
        Holovibes::instance().get_cd().x_cuts = x;

    if (y < frame_descriptor.height)
        Holovibes::instance().get_cd().y_cuts = y;
}

// VALID
void set_q(int value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().q_index = value;
}

// VALID
void set_q_accu(bool is_q_accu, uint q_value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().q_acc_enabled = is_q_accu;
    Holovibes::instance().get_cd().q_acc_level = q_value;
    pipe_refresh();
}

// VALID
void set_p(int value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().pindex = value;
    pipe_refresh();
}

// VALID
void set_composite_intervals(uint composite_p_red, uint composite_p_blue)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_red = composite_p_red;
    Holovibes::instance().get_cd().composite_p_blue = composite_p_blue;
    pipe_refresh();
}

// VALID
void set_composite_intervals_hsv_h_min(uint composite_p_min_h)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_min_h = composite_p_min_h;
    pipe_refresh();
}

// VALID
void set_composite_intervals_hsv_h_max(uint composite_p_max_h)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_max_h = composite_p_max_h;
    pipe_refresh();
}

// VALID
void set_composite_intervals_hsv_s_min(uint composite_p_min_s)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_min_s = composite_p_min_s;
    pipe_refresh();
}

// VALID
void set_composite_intervals_hsv_s_max(uint composite_p_max_s)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_max_s = composite_p_max_s;
    pipe_refresh();
}

// VALID
void set_composite_intervals_hsv_v_min(uint composite_p_min_v)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_min_v = composite_p_min_v;
    pipe_refresh();
}

// VALID
void set_composite_intervals_hsv_v_max(uint composite_p_max_v)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_max_v = composite_p_max_v;
    pipe_refresh();
}

// VALID
void set_composite_weights(uint weight_r, uint weight_g, uint weight_b)
{
    LOG_INFO;
    Holovibes::instance().get_cd().weight_r = weight_r;
    Holovibes::instance().get_cd().weight_g = weight_g;
    Holovibes::instance().get_cd().weight_b = weight_b;
    pipe_refresh();
}

// VALID
void set_composite_auto_weights(bool value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_auto_weights_ = value;
}

// VALID
void select_composite_rgb()
{
    LOG_INFO;

    Holovibes::instance().get_cd().composite_kind = CompositeKind::RGB;
}

// VALID
void select_composite_hsv()
{
    LOG_INFO;

    Holovibes::instance().get_cd().composite_kind = CompositeKind::HSV;
}

// VALID
void actualize_frequency_channel_s(bool composite_p_activated_s)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_activated_s = composite_p_activated_s;
}

// VALID
void actualize_frequency_channel_v(bool composite_p_activated_v)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_activated_v = composite_p_activated_v;
}

// VALID
void actualize_selection_h_gaussian_blur(bool h_blur_activated)
{
    LOG_INFO;
    Holovibes::instance().get_cd().h_blur_activated = h_blur_activated;
}

// VALID
void actualize_kernel_size_blur(uint h_blur_kernel_size)
{
    LOG_INFO;
    Holovibes::instance().get_cd().h_blur_kernel_size = h_blur_kernel_size;
}

// VALID
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

// VALID
void increment_p()
{
    LOG_INFO;

    Holovibes::instance().get_cd().pindex++;
}

// VALID
void decrement_p()
{
    LOG_INFO;

    Holovibes::instance().get_cd().pindex--;
}

// VALID
void set_wavelength(const double value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().lambda = static_cast<float>(value) * 1.0e-9f;

    pipe_refresh();
}

// VALID
void set_z(const double value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().zdistance = static_cast<float>(value);

    pipe_refresh();
}

// VALID
void increment_z()
{
    LOG_INFO;

    set_z(Holovibes::instance().get_cd().zdistance + UserInterfaceDescriptor::instance().z_step_);
}

// VALID
void decrement_z()
{
    LOG_INFO;

    set_z(Holovibes::instance().get_cd().zdistance - UserInterfaceDescriptor::instance().z_step_);
}

// VALID
void set_z_step(const double value)
{
    LOG_INFO;
    UserInterfaceDescriptor::instance().z_step_ = value;
}

// VALID
void set_space_transformation(const std::string& value)
{
    LOG_INFO;

    if (value == "None")
        Holovibes::instance().get_cd().space_transformation = SpaceTransformation::None;
    else if (value == "1FFT")
        Holovibes::instance().get_cd().space_transformation = SpaceTransformation::FFT1;
    else if (value == "2FFT")
        Holovibes::instance().get_cd().space_transformation = SpaceTransformation::FFT2;
    else
    {
        // Shouldn't happen
        Holovibes::instance().get_cd().space_transformation = SpaceTransformation::None;
        LOG_ERROR << "Unknown space transform: " << value << ", falling back to None";
    }
}

// VALID
void set_time_transformation(const std::string& value)
{
    LOG_INFO;

    if (value == "STFT")
        Holovibes::instance().get_cd().time_transformation = TimeTransformation::STFT;
    else if (value == "PCA")
        Holovibes::instance().get_cd().time_transformation = TimeTransformation::PCA;
    else if (value == "None")
        Holovibes::instance().get_cd().time_transformation = TimeTransformation::NONE;
    else if (value == "SSA_STFT")
        Holovibes::instance().get_cd().time_transformation = TimeTransformation::SSA_STFT;
}

// VALID
void adapt_time_transformation_stride_to_batch_size()
{
    if (Holovibes::instance().get_cd().time_transformation_stride < Holovibes::instance().get_cd().batch_size)
        Holovibes::instance().get_cd().time_transformation_stride = Holovibes::instance().get_cd().batch_size.load();
    // Go to lower multiple
    if (Holovibes::instance().get_cd().time_transformation_stride % Holovibes::instance().get_cd().batch_size != 0)
        Holovibes::instance().get_cd().time_transformation_stride -=
            Holovibes::instance().get_cd().time_transformation_stride % Holovibes::instance().get_cd().batch_size;
}

// VALID
void set_unwrapping_2d(const bool value)
{
    LOG_INFO;

    Holovibes::instance().get_compute_pipe()->request_unwrapping_2d(value);

    pipe_refresh();
}

// VALID
void set_accumulation(bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().set_accumulation(Holovibes::instance().get_cd().current_window, value);

    pipe_refresh();
}

// VALID
void set_accumulation_level(int value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().set_accumulation_level(Holovibes::instance().get_cd().current_window, value);

    pipe_refresh();
}

// VALID
void set_composite_area()
{
    LOG_INFO;
    UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::CompositeArea>();
}

// VALID
void set_computation_mode(const Computation computation)
{
    LOG_INFO;

    Holovibes::instance().get_cd().compute_mode = computation;
}

// VALID
void close_critical_compute()
{
    LOG_INFO;
    if (Holovibes::instance().get_cd().convolution_enabled)
        unset_convolution_mode();

    if (Holovibes::instance().get_cd().time_transformation_cuts_enabled)
        cancel_time_transformation_cuts([]() { return; });

    Holovibes::instance().stop_compute();
}

// VALID
void stop_all_worker_controller()
{
    LOG_INFO;

    Holovibes::instance().stop_all_worker_controller();
}

// VALID
bool get_img_acc_slice_enabled()
{
    LOG_INFO;

    return Holovibes::instance().get_cd().get_img_acc_slice_enabled(api::get_current_window());
}

// VALID
unsigned get_img_acc_slice_level()
{
    LOG_INFO;

    return Holovibes::instance().get_cd().get_img_acc_slice_level(api::get_current_window());
}

// VALID
int get_gpu_input_queue_fd_width()
{
    LOG_INFO;

    return Holovibes::instance().get_gpu_input_queue()->get_fd().width;
}

// VALID
int get_gpu_input_queue_fd_height()
{
    LOG_INFO;

    return Holovibes::instance().get_gpu_input_queue()->get_fd().height;
}

// VALID
float get_boundary()
{
    LOG_INFO;

    return Holovibes::instance().get_boundary();
}

#pragma endregion

#pragma region Texture

// VALID
void rotateTexture()
{
    LOG_INFO;

    const WindowKind curWin = Holovibes::instance().get_cd().current_window;

    if (curWin == WindowKind::XYview)
    {
        UserInterfaceDescriptor::instance().displayAngle =
            (UserInterfaceDescriptor::instance().displayAngle == 270.f)
                ? 0.f
                : UserInterfaceDescriptor::instance().displayAngle + 90.f;
        UserInterfaceDescriptor::instance().mainDisplay->setAngle(UserInterfaceDescriptor::instance().displayAngle);
    }
    else if (UserInterfaceDescriptor::instance().sliceXZ && curWin == WindowKind::XZview)
    {
        UserInterfaceDescriptor::instance().xzAngle = (UserInterfaceDescriptor::instance().xzAngle == 270.f)
                                                          ? 0.f
                                                          : UserInterfaceDescriptor::instance().xzAngle + 90.f;
        UserInterfaceDescriptor::instance().sliceXZ->setAngle(UserInterfaceDescriptor::instance().xzAngle);
    }
    else if (UserInterfaceDescriptor::instance().sliceYZ && curWin == WindowKind::YZview)
    {
        UserInterfaceDescriptor::instance().yzAngle = (UserInterfaceDescriptor::instance().yzAngle == 270.f)
                                                          ? 0.f
                                                          : UserInterfaceDescriptor::instance().yzAngle + 90.f;
        UserInterfaceDescriptor::instance().sliceYZ->setAngle(UserInterfaceDescriptor::instance().yzAngle);
    }
}

// VALID
void flipTexture()
{
    LOG_INFO;

    const WindowKind curWin = Holovibes::instance().get_cd().current_window;

    if (curWin == WindowKind::XYview)
    {
        UserInterfaceDescriptor::instance().displayFlip = !UserInterfaceDescriptor::instance().displayFlip;
        UserInterfaceDescriptor::instance().mainDisplay->setFlip(UserInterfaceDescriptor::instance().displayFlip);
    }
    else if (UserInterfaceDescriptor::instance().sliceXZ && curWin == WindowKind::XZview)
    {
        UserInterfaceDescriptor::instance().xzFlip = !UserInterfaceDescriptor::instance().xzFlip;
        UserInterfaceDescriptor::instance().sliceXZ->setFlip(UserInterfaceDescriptor::instance().xzFlip);
    }
    else if (UserInterfaceDescriptor::instance().sliceYZ && curWin == WindowKind::YZview)
    {
        UserInterfaceDescriptor::instance().yzFlip = !UserInterfaceDescriptor::instance().yzFlip;
        UserInterfaceDescriptor::instance().sliceYZ->setFlip(UserInterfaceDescriptor::instance().yzFlip);
    }
}

#pragma endregion

#pragma region Contrast - Log

// VALID
void set_contrast_mode(bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().contrast_enabled = value;
    Holovibes::instance().get_cd().contrast_auto_refresh = true;
    pipe_refresh();
}

// VALID
void set_auto_contrast_cuts()
{
    LOG_INFO;

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XZview);
        pipe->autocontrast_end_pipe(WindowKind::YZview);
    }
}

// VALID
bool set_auto_contrast()
{
    LOG_INFO;

    try
    {
        if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
        {
            pipe->autocontrast_end_pipe(Holovibes::instance().get_cd().current_window);
            return true;
        }
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what() << std::endl;
    }

    return false;
}

// VALID
void set_auto_contrast_all()
{
    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XYview);
        if (Holovibes::instance().get_cd().time_transformation_cuts_enabled)
        {
            pipe->autocontrast_end_pipe(WindowKind::XZview);
            pipe->autocontrast_end_pipe(WindowKind::YZview);
        }
        if (Holovibes::instance().get_cd().filter2d_view_enabled)
            pipe->autocontrast_end_pipe(WindowKind::Filter2D);

        pipe_refresh();
    }
}

// VALID
void set_contrast_min(const double value)
{
    LOG_INFO;

    // Get the minimum contrast value rounded for the comparison
    const float old_val =
        Holovibes::instance().get_cd().get_truncate_contrast_min(Holovibes::instance().get_cd().current_window);
    // Floating number issue: cast to float for the comparison
    const float val = value;
    if (old_val != val)
    {
        Holovibes::instance().get_cd().set_contrast_min(Holovibes::instance().get_cd().current_window, value);
        pipe_refresh();
    }
}

// VALID
void set_contrast_max(const double value)
{
    LOG_INFO;

    // Get the maximum contrast value rounded for the comparison
    const float old_val =
        Holovibes::instance().get_cd().get_truncate_contrast_max(Holovibes::instance().get_cd().current_window);
    // Floating number issue: cast to float for the comparison
    const float val = value;
    if (old_val != val)
    {
        Holovibes::instance().get_cd().set_contrast_max(Holovibes::instance().get_cd().current_window, value);
        pipe_refresh();
    }
}

// VALID
void invert_contrast(bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().contrast_invert = value;
    pipe_refresh();
}

// VALID
void set_auto_refresh_contrast(bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().contrast_auto_refresh = value;
    pipe_refresh();
}

// VALID
void set_log_scale(const bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().set_log_scale_slice_enabled(Holovibes::instance().get_cd().current_window, value);
    if (value && Holovibes::instance().get_cd().contrast_enabled)
        set_auto_contrast();

    pipe_refresh();
}

// VALID
float get_contrast_min()
{
    LOG_INFO;

    return Holovibes::instance().get_cd().get_contrast_min(api::get_current_window());
}

// VALID
float get_contrast_max()
{
    LOG_INFO;

    return Holovibes::instance().get_cd().get_contrast_max(api::get_current_window());
}

// VALID
bool get_img_log_scale_slice_enabled()
{
    LOG_INFO;

    return Holovibes::instance().get_cd().get_img_log_scale_slice_enabled(api::get_current_window());
}

#pragma endregion

#pragma region Convolution

// VALID
void update_convo_kernel(const std::string& value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().set_convolution(true, value);

    try
    {
        auto pipe = Holovibes::instance().get_compute_pipe();
        pipe->request_convolution();
        // Wait for the convolution to be enabled for notify
        while (pipe->get_convolution_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        Holovibes::instance().get_cd().convolution_enabled = false;
        LOG_ERROR << e.what();
    }
}

// VALID
void set_convolution_mode(std::string& str)
{
    LOG_INFO;

    Holovibes::instance().get_cd().set_convolution(true, str);

    try
    {
        auto pipe = Holovibes::instance().get_compute_pipe();

        pipe->request_convolution();
        // Wait for the convolution to be enabled for notify
        while (pipe->get_convolution_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        Holovibes::instance().get_cd().convolution_enabled = false;
        LOG_ERROR << e.what();
    }
}

// VALID
void unset_convolution_mode()
{
    LOG_INFO;

    try
    {
        auto pipe = Holovibes::instance().get_compute_pipe();

        pipe->request_disable_convolution();
        // Wait for the convolution to be disabled for notify
        while (pipe->get_disable_convolution_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        Holovibes::instance().get_cd().convolution_enabled = false;
        LOG_ERROR << e.what();
    }
}

// VALID
void set_divide_convolution_mode(const bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().divide_convolution_enabled = value;

    pipe_refresh();
}

#pragma endregion

#pragma region Reticle

// VALID
void display_reticle(bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().reticle_enabled = value;
    if (value)
    {
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Reticle>();
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_default();
    }
    else
    {
        UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(gui::Reticle);
    }

    pipe_refresh();
}

// VALID
void reticle_scale(double value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().reticle_scale = value;
    pipe_refresh();
}

#pragma endregion

#pragma region Chart

// VALID
void activeNoiseZone()
{
    LOG_INFO;

    UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Noise>();
}

// VALID
void activeSignalZone()
{
    LOG_INFO;
    UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<gui::Signal>();
}

// VALID
void start_chart_display()
{
    LOG_INFO;

    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_display_chart();

    // Wait for the chart display to be enabled for notify
    while (pipe->get_chart_display_requested())
        continue;

    UserInterfaceDescriptor::instance().plot_window_ =
        std::make_unique<gui::PlotWindow>(*Holovibes::instance().get_compute_pipe()->get_chart_display_queue(),
                                          UserInterfaceDescriptor::instance().auto_scale_point_threshold_,
                                          "Chart");
}

// VALID
void stop_chart_display()
{
    LOG_INFO;

    try
    {
        auto pipe = Holovibes::instance().get_compute_pipe();
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

// VALID
void set_record_frame_step(int value) { UserInterfaceDescriptor::instance().record_frame_step_ = value; }

// VALID
const std::string browse_record_output_file(std::string& std_filepath)
{
    LOG_INFO;

    // FIXME: path separator should depend from system
    std::replace(std_filepath.begin(), std_filepath.end(), '/', '\\');
    std::filesystem::path path = std::filesystem::path(std_filepath);

    // FIXME Opti: we could be all these 3 operations below on a single string processing
    UserInterfaceDescriptor::instance().record_output_directory_ = path.parent_path().string();
    const std::string file_ext = path.extension().string();
    UserInterfaceDescriptor::instance().default_output_filename_ = path.stem().string();

    return file_ext;
}

// VALID
void set_record_mode(const std::string& text)
{
    LOG_INFO;

    if (text == "Chart")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::CHART;
    else if (text == "Processed Image")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::HOLOGRAM;
    else if (text == "Raw Image")
        UserInterfaceDescriptor::instance().record_mode_ = RecordMode::RAW;
    else
        throw std::exception("Record mode not handled");
}

// VALID
bool start_record_preconditions(const bool batch_enabled,
                                const bool nb_frame_checked,
                                std::optional<unsigned int> nb_frames_to_record,
                                const std::string& batch_input_path)
{
    LOG_INFO;
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

// VALID
void start_record(const bool batch_enabled,
                  std::optional<unsigned int> nb_frames_to_record,
                  std::string& output_path,
                  std::string& batch_input_path,
                  std::function<void()> callback)
{
    LOG_INFO;

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
        else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::HOLOGRAM)
        {
            Holovibes::instance().start_frame_record(output_path, nb_frames_to_record, false, 0, callback);
        }
        else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::RAW)
        {
            Holovibes::instance().start_frame_record(output_path, nb_frames_to_record, true, 0, callback);
        }
    }
}

// VALID
void stop_record()
{
    LOG_INFO;
    Holovibes::instance().stop_batch_gpib();

    if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
        Holovibes::instance().stop_chart_record();
    else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::HOLOGRAM ||
             UserInterfaceDescriptor::instance().record_mode_ == RecordMode::RAW)
        Holovibes::instance().stop_frame_record();
}

// VALID
void record_finished()
{
    LOG_INFO;

    UserInterfaceDescriptor::instance().is_recording_ = false;
}

#pragma endregion

#pragma region Import

// VALID
void import_stop()
{
    LOG_INFO;

    Holovibes::instance().stop_all_worker_controller();
    Holovibes::instance().start_information_display(false);

    close_critical_compute();

    Holovibes::instance().get_cd().is_computation_stopped = true;
}

// VALID
bool import_start(
    std::string& file_path, unsigned int fps, size_t first_frame, bool load_file_in_gpu, size_t last_frame)
{
    LOG_INFO;

    Holovibes::instance().get_cd().is_computation_stopped = false;
    // Gather all the usefull data from the ui import panel
    return init_holovibes_import_mode(file_path, fps, first_frame, load_file_in_gpu, last_frame);
}

// VALID
std::optional<io_files::InputFrameFile*> import_file(const std::string& filename)
{
    LOG_INFO;

    if (!filename.empty())
    {

        // Will throw if the file format (extension) cannot be handled
        auto input_file = io_files::InputFrameFileFactory::open(filename);

        return input_file;
    }

    return std::nullopt;
}

#pragma endregion

#pragma region Information

void start_information_display(bool is_cli, const std::function<void()>& callback)
{
    LOG_INFO;

    Holovibes::instance().start_information_display(false, callback);
}

void set_display_info_text_function(const std::function<void(const std::string&)>& display_info_text_fun)
{
    LOG_INFO;

    Holovibes::instance().get_info_container().set_display_info_text_function(display_info_text_fun);
}

void set_update_progress_function(
    const std::function<void(InformationContainer::ProgressType, size_t, size_t)>& function)
{
    LOG_INFO;

    Holovibes::instance().get_info_container().set_update_progress_function(function);
}

#pragma endregion

} // namespace holovibes::api
