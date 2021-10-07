#include "api.hh"

#include <optional>
namespace holovibes::api
{

#pragma region Local

void open_file(const std::string& path)
{
    LOG_INFO;
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString(path.c_str())));
}

void pipe_refresh(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
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

bool init_holovibes_import_mode(UserInterfaceDescriptor& ui_descriptor,
                                std::string& file_path,
                                unsigned int fps,
                                size_t first_frame,
                                bool load_file_in_gpu,
                                size_t last_frame)
{
    LOG_INFO;

    // Set the image rendering ui params
    Holovibes::instance().get_cd().time_transformation_stride = std::ceil(static_cast<float>(fps) / 20.0f);
    Holovibes::instance().get_cd().batch_size = 1;

    // Because we are in import mode
    ui_descriptor.is_enabled_camera_ = false;

    try
    {

        Holovibes::instance().init_input_queue(ui_descriptor.file_fd_);
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
        ui_descriptor.is_enabled_camera_ = false;
        Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();
        return false;
    }
    ui_descriptor.is_enabled_camera_ = true;
    return true;
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

bool is_raw_mode(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    return Holovibes::instance().get_cd().compute_mode == Computation::Raw;
}

void remove_infos()
{
    LOG_INFO;
    Holovibes::instance().get_info_container().clear();
}

void close_windows(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    ui_descriptor.sliceXZ.reset(nullptr);
    ui_descriptor.sliceYZ.reset(nullptr);

    ui_descriptor.plot_window_.reset(nullptr);
    ui_descriptor.mainDisplay.reset(nullptr);

    ui_descriptor.lens_window.reset(nullptr);
    Holovibes::instance().get_cd().gpu_lens_display_enabled = false;

    ui_descriptor.filter2d_window.reset(nullptr);
    Holovibes::instance().get_cd().filter2d_view_enabled = false;

    /* Raw view & recording */
    ui_descriptor.raw_window.reset(nullptr);
    Holovibes::instance().get_cd().raw_view_enabled = false;

    // Disable overlays
    Holovibes::instance().get_cd().reticle_enabled = false;
}

#pragma endregion

#pragma region Ini
void configure_holovibes()
{
    LOG_INFO;
    open_file(::holovibes::ini::get_global_ini_path());
}

void write_ini(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    mainwindow.write_ini((QString) "");
}

void write_ini(::holovibes::gui::MainWindow& mainwindow,
               UserInterfaceDescriptor& ui_descriptor,
               const std::string& filename)
{
    LOG_INFO;

    // boost::property_tree::ptree ptree;
    // Saves the current state of holovibes in holovibes.ini located in Holovibes.exe directory
    mainwindow.save_ini(filename.empty() ? ::holovibes::ini::get_global_ini_path() : filename);
}

void browse_export_ini(::holovibes::gui::MainWindow& mainwindow,
                       UserInterfaceDescriptor& ui_descriptor,
                       const std::string& filename)
{
    LOG_INFO;

    mainwindow.write_ini(filename);
}

void browse_import_ini(::holovibes::gui::MainWindow& mainwindow,
                       UserInterfaceDescriptor& ui_descriptor,
                       const std::string& filename)
{
    LOG_INFO;

    ::holovibes::api::reload_ini(mainwindow, ui_descriptor, filename);
}

void reload_ini(::holovibes::gui::MainWindow& mainwindow,
                UserInterfaceDescriptor& ui_descriptor,
                const std::string& filename)
{
    LOG_INFO;
    mainwindow.import_stop();
    try
    {
        mainwindow.load_ini(filename.empty() ? ::holovibes::ini::get_global_ini_path() : filename);
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        LOG_INFO << e.what() << std::endl;
    }
    if (ui_descriptor.import_type_ == ::holovibes::UserInterfaceDescriptor::ImportType::File)
    {
        mainwindow.import_start();
    }
    else if (ui_descriptor.import_type_ == ::holovibes::UserInterfaceDescriptor::ImportType::Camera)
    {
        mainwindow.change_camera(ui_descriptor.kCamera);
    }
}

void reload_ini(::holovibes::gui::MainWindow& mainwindow)
{
    LOG_INFO;
    mainwindow.reload_ini("");
}

void load_ini(::holovibes::gui::MainWindow& mainwindow,
              UserInterfaceDescriptor& ui_descriptor,
              const std::string& path,
              boost::property_tree::ptree& ptree)
{
    LOG_INFO;

    boost::property_tree::ini_parser::read_ini(path, ptree);

    if (!ptree.empty())
    {
        // Load general compute data
        ini::load_ini(ptree, Holovibes::instance().get_cd());

        // Load window specific data
        ui_descriptor.default_output_filename_ =
            ptree.get<std::string>("files.default_output_filename", ui_descriptor.default_output_filename_);
        ui_descriptor.record_output_directory_ =
            ptree.get<std::string>("files.record_output_directory", ui_descriptor.record_output_directory_);
        ui_descriptor.file_input_directory_ =
            ptree.get<std::string>("files.file_input_directory", ui_descriptor.file_input_directory_);
        ui_descriptor.batch_input_directory_ =
            ptree.get<std::string>("files.batch_input_directory", ui_descriptor.batch_input_directory_);

        const float z_step = ptree.get<float>("image_rendering.z_step", ui_descriptor.z_step_);
        if (z_step > 0.0f)
            mainwindow.set_z_step(z_step);

        ui_descriptor.last_img_type_ = Holovibes::instance().get_cd().img_type == ImgType::Composite
                                           ? "Composite image"
                                           : ui_descriptor.last_img_type_;

        ui_descriptor.displayAngle = ptree.get("view.mainWindow_rotate", ui_descriptor.displayAngle);
        ui_descriptor.xzAngle = ptree.get<float>("view.xCut_rotate", ui_descriptor.xzAngle);
        ui_descriptor.yzAngle = ptree.get<float>("view.yCut_rotate", ui_descriptor.yzAngle);
        ui_descriptor.displayFlip = ptree.get("view.mainWindow_flip", ui_descriptor.displayFlip);
        ui_descriptor.xzFlip = ptree.get("view.xCut_flip", ui_descriptor.xzFlip);
        ui_descriptor.yzFlip = ptree.get("view.yCut_flip", ui_descriptor.yzFlip);

        ui_descriptor.auto_scale_point_threshold_ =
            ptree.get<size_t>("chart.auto_scale_point_threshold", ui_descriptor.auto_scale_point_threshold_);

        const uint record_frame_step = ptree.get<uint>("record.record_frame_step", ui_descriptor.record_frame_step_);
        mainwindow.set_record_frame_step(record_frame_step);

        ui_descriptor.window_max_size = ptree.get<uint>("display.main_window_max_size", 768);
        ui_descriptor.time_transformation_cuts_window_max_size =
            ptree.get<uint>("display.time_transformation_cuts_window_max_size", 512);
        ui_descriptor.auxiliary_window_max_size = ptree.get<uint>("display.auxiliary_window_max_size", 512);
    }
}

void save_ini(UserInterfaceDescriptor& ui_descriptor, const std::string& path, boost::property_tree::ptree& ptree)
{
    LOG_INFO;

    // Save general compute data
    ini::save_ini(ptree, Holovibes::instance().get_cd());

    // Save window specific data
    ptree.put<std::string>("files.default_output_filename", ui_descriptor.default_output_filename_);
    ptree.put<std::string>("files.record_output_directory", ui_descriptor.record_output_directory_);
    ptree.put<std::string>("files.file_input_directory", ui_descriptor.file_input_directory_);
    ptree.put<std::string>("files.batch_input_directory", ui_descriptor.batch_input_directory_);

    ptree.put<int>("image_rendering.camera", static_cast<int>(ui_descriptor.kCamera));

    ptree.put<double>("image_rendering.z_step", ui_descriptor.z_step_);

    ptree.put<float>("view.mainWindow_rotate", ui_descriptor.displayAngle);
    ptree.put<float>("view.xCut_rotate", ui_descriptor.xzAngle);
    ptree.put<float>("view.yCut_rotate", ui_descriptor.yzAngle);
    ptree.put<int>("view.mainWindow_flip", ui_descriptor.displayFlip);
    ptree.put<int>("view.xCut_flip", ui_descriptor.xzFlip);
    ptree.put<int>("view.yCut_flip", ui_descriptor.yzFlip);

    ptree.put<size_t>("chart.auto_scale_point_threshold", ui_descriptor.auto_scale_point_threshold_);

    ptree.put<uint>("record.record_frame_step", ui_descriptor.record_frame_step_);

    ptree.put<uint>("display.main_window_max_size", ui_descriptor.window_max_size);
    ptree.put<uint>("display.time_transformation_cuts_window_max_size",
                    ui_descriptor.time_transformation_cuts_window_max_size);
    ptree.put<uint>("display.auxiliary_window_max_size", ui_descriptor.auxiliary_window_max_size);

    boost::property_tree::write_ini(path, ptree);

    LOG_INFO << "Configuration file holovibes.ini overwritten at " << path << std::endl;
}

#pragma endregion

#pragma region Close Compute

void camera_none(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    close_windows(ui_descriptor);
    close_critical_compute(ui_descriptor);
    if (!is_raw_mode(ui_descriptor))
        Holovibes::instance().stop_compute();
    Holovibes::instance().stop_frame_read();
    remove_infos();

    ui_descriptor.is_enabled_camera_ = false;
    Holovibes::instance().get_cd().is_computation_stopped = true;
}

void reset(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    Config& config = global::global_config;
    int device = 0;

    close_critical_compute(ui_descriptor);
    mainwindow.camera_none();

    // TOD: qApp must be strongly related to qt window
    qApp->processEvents();

    if (!is_raw_mode(ui_descriptor))
        Holovibes::instance().stop_compute();
    Holovibes::instance().stop_frame_read();
    Holovibes::instance().get_cd().pindex = 0;
    Holovibes::instance().get_cd().time_transformation_size = 1;
    ui_descriptor.is_enabled_camera_ = false;
    if (config.set_cuda_device)
    {
        if (config.auto_device_number)
        {
            cudaGetDevice(&device);
            config.device_number = device;
        }
        else
            device = config.device_number;
        cudaSetDevice(device);
    }
    cudaDeviceSynchronize();
    cudaDeviceReset();
    close_windows(ui_descriptor);
    remove_infos();
    Holovibes::instance().reload_streams();
    try
    {
        mainwindow.load_ini(::holovibes::ini::get_global_ini_path());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        LOG_WARN << ::holovibes::ini::get_global_ini_path()
                 << ": Config file not found. It will use the default values.";
    }
}

void closeEvent(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    close_windows(ui_descriptor);
    if (!Holovibes::instance().get_cd().is_computation_stopped)
        close_critical_compute(ui_descriptor);
    mainwindow.camera_none();
    remove_infos();
    mainwindow.save_ini(::holovibes::ini::get_global_ini_path());
}

#pragma endregion

#pragma region Cameras

// TODO: we shouldn't use const uint image_mode_index that is a qt drop list concept
bool change_camera(::holovibes::gui::MainWindow& mainwindow,
                   UserInterfaceDescriptor& ui_descriptor,
                   CameraKind c,
                   const uint image_mode_index)
{
    LOG_INFO;

    mainwindow.camera_none();

    bool res = false;

    if (c == CameraKind::NONE)
    {
        return res;
    }

    try
    {
        ui_descriptor.mainDisplay.reset(nullptr);
        if (!is_raw_mode(ui_descriptor))
            Holovibes::instance().stop_compute();
        Holovibes::instance().stop_frame_read();

        set_camera_timeout();

        set_computation_mode(Holovibes::instance(), image_mode_index);

        Holovibes::instance().start_camera_frame_read(c);
        ui_descriptor.is_enabled_camera_ = true;
        set_image_mode(mainwindow, ui_descriptor, true, image_mode_index);
        ui_descriptor.import_type_ = ::holovibes::UserInterfaceDescriptor::ImportType::Camera;
        ui_descriptor.kCamera = c;

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

void camera_ids(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    mainwindow.change_camera(::holovibes::CameraKind::IDS);
}

void camera_phantom(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    mainwindow.change_camera(::holovibes::CameraKind::Phantom);
}

void camera_bitflow_cyton(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    mainwindow.change_camera(::holovibes::CameraKind::BitflowCyton);
}

void camera_hamamatsu(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    mainwindow.change_camera(::holovibes::CameraKind::Hamamatsu);
}

void camera_adimec(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    mainwindow.change_camera(::holovibes::CameraKind::Adimec);
}

void camera_xiq(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    mainwindow.change_camera(::holovibes::CameraKind::xiQ);
}

void camera_xib(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    mainwindow.change_camera(::holovibes::CameraKind::xiB);
}

void configure_camera(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    open_file(std::filesystem::current_path().generic_string() + "/" + Holovibes::instance().get_camera_ini_path());
}

void set_camera_timeout()
{
    LOG_INFO;
    camera::FRAME_TIMEOUT = global::global_config.frame_timeout;
}

#pragma endregion

#pragma region Image Mode

void init_image_mode(UserInterfaceDescriptor& ui_descriptor, QPoint& position, QSize& size)
{
    LOG_INFO;

    if (ui_descriptor.mainDisplay)
    {
        position = ui_descriptor.mainDisplay->framePosition();
        size = ui_descriptor.mainDisplay->size();
        ui_descriptor.mainDisplay.reset(nullptr);
    }
}

bool set_raw_mode(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    close_windows(ui_descriptor);
    close_critical_compute(ui_descriptor);

    if (ui_descriptor.is_enabled_camera_)
    {
        QPoint pos(0, 0);
        const camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();
        unsigned short width = fd.width;
        unsigned short height = fd.height;
        get_good_size(width, height, ui_descriptor.window_max_size);
        QSize size(width, height);
        mainwindow.init_image_mode(pos, size);
        Holovibes::instance().get_cd().compute_mode = Computation::Raw;
        createPipe(mainwindow, ui_descriptor);
        ui_descriptor.mainDisplay.reset(
            new holovibes::gui::RawWindow(pos, size, Holovibes::instance().get_gpu_input_queue().get()));
        ui_descriptor.mainDisplay->setTitle(QString("XY view"));
        ui_descriptor.mainDisplay->setCd(&Holovibes::instance().get_cd());
        ui_descriptor.mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));
        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        Holovibes::instance().get_info_container().add_indication(InformationContainer::IndicationType::INPUT_FORMAT,
                                                                  fd_info);
        unset_convolution_mode(ui_descriptor);
        set_divide_convolution_mode(ui_descriptor, false);

        return true;
    }

    return false;
}

void createPipe(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    try
    {
        Holovibes::instance().start_compute();
        Holovibes::instance().get_compute_pipe()->register_observer(mainwindow);
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot create Pipe: " << e.what();
    }
}

void createHoloWindow(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    QPoint pos(0, 0);
    const camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, ui_descriptor.window_max_size);
    QSize size(width, height);
    mainwindow.init_image_mode(pos, size);
    /* ---------- */
    try
    {
        ui_descriptor.mainDisplay.reset(
            new ::holovibes::gui::HoloWindow(pos,
                                             size,
                                             Holovibes::instance().get_gpu_output_queue().get(),
                                             Holovibes::instance().get_compute_pipe(),
                                             ui_descriptor.sliceXZ,
                                             ui_descriptor.sliceYZ,
                                             &mainwindow));
        ui_descriptor.mainDisplay->set_is_resize(false);
        ui_descriptor.mainDisplay->setTitle(QString("XY view"));
        ui_descriptor.mainDisplay->setCd(&Holovibes::instance().get_cd());
        ui_descriptor.mainDisplay->resetTransform();
        ui_descriptor.mainDisplay->setAngle(ui_descriptor.displayAngle);
        ui_descriptor.mainDisplay->setFlip(ui_descriptor.displayFlip);
        ui_descriptor.mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "createHoloWindow: " << e.what();
    }
}

bool set_holographic_mode(::holovibes::gui::MainWindow& mainwindow,
                          UserInterfaceDescriptor& ui_descriptor,
                          camera::FrameDescriptor& fd)
{
    LOG_INFO;
    // That function is used to reallocate the buffers since the Square
    // input mode could have changed
    /* Close windows & destory thread compute */
    close_windows(ui_descriptor);
    close_critical_compute(ui_descriptor);

    /* ---------- */
    try
    {
        Holovibes::instance().get_cd().compute_mode = Computation::Hologram;
        /* Pipe & Window */
        mainwindow.createPipe();
        mainwindow.createHoloWindow();
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

void refreshViewMode(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, uint index)
{
    LOG_INFO;
    float old_scale = 1.f;
    glm::vec2 old_translation(0.f, 0.f);
    if (ui_descriptor.mainDisplay)
    {
        old_scale = ui_descriptor.mainDisplay->getScale();
        old_translation = ui_descriptor.mainDisplay->getTranslate();
    }

    close_windows(ui_descriptor);
    close_critical_compute(ui_descriptor);

    Holovibes::instance().get_cd().img_type = static_cast<ImgType>(index);

    try
    {
        mainwindow.createPipe();
        mainwindow.createHoloWindow();
        ui_descriptor.mainDisplay->setScale(old_scale);
        ui_descriptor.mainDisplay->setTranslate(old_translation[0], old_translation[1]);
    }
    catch (const std::runtime_error& e)
    {
        ui_descriptor.mainDisplay.reset(nullptr);
        LOG_ERROR << "refreshViewMode: " << e.what();
    }
}

void set_view_mode(::holovibes::gui::MainWindow& mainwindow,
                   UserInterfaceDescriptor& ui_descriptor,
                   const std::string& value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return;

    if (mainwindow.need_refresh(ui_descriptor.last_img_type_, value))
    {
        mainwindow.refreshViewMode();
        if (Holovibes::instance().get_cd().img_type == ImgType::Composite)
        {
            mainwindow.set_composite_values();
        }
    }
    ui_descriptor.last_img_type_ = value;

    auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get());

    pipe->insert_fn_end_vect(mainwindow.get_view_mode_callback());
    pipe_refresh(ui_descriptor);

    // Force XYview autocontrast
    pipe->autocontrast_end_pipe(WindowKind::XYview);
    // Force cuts views autocontrast if needed
    set_auto_contrast_cuts(ui_descriptor);
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
    else if (Holovibes::instance().get_cd().compute_mode == Computation::Raw)
        mainwindow.set_raw_mode();
    else if (Holovibes::instance().get_cd().compute_mode == Computation::Hologram)
        mainwindow.set_holographic_mode();
}

#pragma endregion

#pragma region Batch

void update_batch_size(UserInterfaceDescriptor& ui_descriptor, std::function<void()> callback, const uint batch_size)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return;

    if (batch_size == Holovibes::instance().get_cd().batch_size)
        return;

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
    {
        pipe->insert_fn_end_vect(callback);
    }
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

#pragma endregion

#pragma region STFT

void update_time_transformation_stride(UserInterfaceDescriptor& ui_descriptor,
                                       std::function<void()> callback,
                                       const uint time_transformation_stride)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return;

    if (time_transformation_stride == Holovibes::instance().get_cd().time_transformation_stride)
        return;

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
    {
        pipe->insert_fn_end_vect(callback);
    }
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

bool toggle_time_transformation_cuts(::holovibes::gui::MainWindow& mainwindow,
                                     UserInterfaceDescriptor& ui_descriptor,
                                     const bool checked)
{
    LOG_INFO;

    if (!checked)
    {
        mainwindow.cancel_time_transformation_cuts();
        return false;
    }

    // if checked
    try
    {
        Holovibes::instance().get_compute_pipe()->create_stft_slice_queue();
        // set positions of new windows according to the position of the
        // main GL window
        QPoint xzPos = ui_descriptor.mainDisplay->framePosition() + QPoint(0, ui_descriptor.mainDisplay->height() + 42);
        QPoint yzPos = ui_descriptor.mainDisplay->framePosition() + QPoint(ui_descriptor.mainDisplay->width() + 20, 0);
        const ushort nImg = Holovibes::instance().get_cd().time_transformation_size;
        uint time_transformation_size = std::max(256u, std::min(512u, (uint)nImg));

        if (time_transformation_size > ui_descriptor.time_transformation_cuts_window_max_size)
            time_transformation_size = ui_descriptor.time_transformation_cuts_window_max_size;

        while (Holovibes::instance().get_compute_pipe()->get_update_time_transformation_size_request())
            continue;
        while (Holovibes::instance().get_compute_pipe()->get_cuts_request())
            continue;

        ui_descriptor.sliceXZ.reset(
            new ::holovibes::gui::SliceWindow(xzPos,
                                              QSize(ui_descriptor.mainDisplay->width(), time_transformation_size),
                                              Holovibes::instance().get_compute_pipe()->get_stft_slice_queue(0).get(),
                                              ::holovibes::gui::KindOfView::SliceXZ,
                                              &mainwindow));
        ui_descriptor.sliceXZ->setTitle("XZ view");
        ui_descriptor.sliceXZ->setAngle(ui_descriptor.xzAngle);
        ui_descriptor.sliceXZ->setFlip(ui_descriptor.xzFlip);
        ui_descriptor.sliceXZ->setCd(&Holovibes::instance().get_cd());

        ui_descriptor.sliceYZ.reset(
            new ::holovibes::gui::SliceWindow(yzPos,
                                              QSize(time_transformation_size, ui_descriptor.mainDisplay->height()),
                                              Holovibes::instance().get_compute_pipe()->get_stft_slice_queue(1).get(),
                                              ::holovibes::gui::KindOfView::SliceYZ,
                                              &mainwindow));
        ui_descriptor.sliceYZ->setTitle("YZ view");
        ui_descriptor.sliceYZ->setAngle(ui_descriptor.yzAngle);
        ui_descriptor.sliceYZ->setFlip(ui_descriptor.yzFlip);
        ui_descriptor.sliceYZ->setCd(&Holovibes::instance().get_cd());

        ui_descriptor.mainDisplay->getOverlayManager().create_overlay<::holovibes::gui::Cross>();
        Holovibes::instance().get_cd().time_transformation_cuts_enabled = true;
        set_auto_contrast_cuts(ui_descriptor);
        auto holo = dynamic_cast<::holovibes::gui::HoloWindow*>(ui_descriptor.mainDisplay.get());
        if (holo)
            holo->update_slice_transforms();
        return true;
    }
    catch (const std::logic_error& e)
    {
        LOG_ERROR << e.what() << std::endl;
        mainwindow.cancel_time_transformation_cuts();
    }

    return false;
}

bool cancel_time_transformation_cuts(UserInterfaceDescriptor& ui_descriptor, std::function<void()> callback)
{
    LOG_INFO;

    if (!Holovibes::instance().get_cd().time_transformation_cuts_enabled)
    {
        return false;
    }

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

#pragma endregion

#pragma region Computation

void change_window(UserInterfaceDescriptor& ui_descriptor, const int index)
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

    pipe_refresh(ui_descriptor);
}

void toggle_renormalize(UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().renorm_enabled = value;
    Holovibes::instance().get_compute_pipe()->request_clear_img_acc();

    pipe_refresh(ui_descriptor);
}

bool set_filter2d(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, bool checked)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor))
        return false;

    if (checked)
    {
        if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
            pipe->autocontrast_end_pipe(WindowKind::XYview);
        Holovibes::instance().get_cd().filter2d_enabled = checked;
    }
    else
    {
        Holovibes::instance().get_cd().filter2d_enabled = checked;
        mainwindow.cancel_filter2d();
    }

    pipe_refresh(ui_descriptor);
    return true;
}

void disable_filter2d_view(UserInterfaceDescriptor& ui_descriptor, const int index)
{
    LOG_INFO;

    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_disable_filter2d_view();

    // Wait for the filter2d view to be disabled for notify
    while (pipe->get_disable_filter2d_view_requested())
        continue;

    // Change the focused window
    change_window(ui_descriptor, index);
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
            auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get());
            if (pipe)
            {
                pipe->request_filter2d_view();

                const camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_output_queue()->get_fd();
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
                ui_descriptor.filter2d_window->setCd(&Holovibes::instance().get_cd());

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
        ui_descriptor.filter2d_window.reset(nullptr);
        res = false;
    }

    pipe_refresh(ui_descriptor);
    return res;
}

bool set_filter2d_n2(UserInterfaceDescriptor& ui_descriptor, int n)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    Holovibes::instance().get_cd().filter2d_n2 = n;

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
    }

    pipe_refresh(ui_descriptor);
    return true;
}

bool set_filter2d_n1(UserInterfaceDescriptor& ui_descriptor, int n)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    Holovibes::instance().get_cd().filter2d_n1 = n;

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
    }

    pipe_refresh(ui_descriptor);
    return true;
}

bool cancel_filter2d(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    if (Holovibes::instance().get_cd().filter2d_view_enabled)
        mainwindow.update_filter2d_view(false);

    pipe_refresh(ui_descriptor);

    return true;
}

void set_fft_shift(UserInterfaceDescriptor& ui_descriptor, const bool value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return;

    Holovibes::instance().get_cd().fft_shift_enabled = value;
    pipe_refresh(ui_descriptor);
}

bool set_time_transformation_size(UserInterfaceDescriptor& ui_descriptor,
                                  int time_transformation_size,
                                  std::function<void()> callback)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    time_transformation_size = std::max(1, time_transformation_size);

    if (time_transformation_size == Holovibes::instance().get_cd().time_transformation_size)
        return false;

    auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect(callback);
    }

    return true;
}

std::optional<bool>
update_lens_view(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    std::optional<bool> res = true;

    Holovibes::instance().get_cd().gpu_lens_display_enabled = value;

    if (value)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos =
                ui_descriptor.mainDisplay->framePosition() + QPoint(ui_descriptor.mainDisplay->width() + 310, 0);
            ICompute* pipe = Holovibes::instance().get_compute_pipe().get();

            const ::camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();
            ushort lens_window_width = fd.width;
            ushort lens_window_height = fd.height;
            get_good_size(lens_window_width, lens_window_height, ui_descriptor.auxiliary_window_max_size);

            ui_descriptor.lens_window.reset(
                new ::holovibes::gui::RawWindow(pos,
                                                QSize(lens_window_width, lens_window_height),
                                                pipe->get_lens_queue().get(),
                                                ::holovibes::gui::KindOfView::Lens));

            ui_descriptor.lens_window->setTitle("Lens view");
            ui_descriptor.lens_window->setCd(&Holovibes::instance().get_cd());
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

    Holovibes::instance().get_cd().gpu_lens_display_enabled = false;
    Holovibes::instance().get_compute_pipe()->request_disable_lens_view();
}

std::optional<bool>
update_raw_view(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    std::optional<bool> res = true;

    if (value)
    {
        if (Holovibes::instance().get_cd().batch_size > global::global_config.output_queue_max_size)
        {
            LOG_ERROR << "[RAW VIEW] Batch size must be lower than output queue size";
            return std::nullopt;
        }

        auto pipe = Holovibes::instance().get_compute_pipe();
        pipe->request_raw_view();

        // Wait for the raw view to be enabled for notify
        while (pipe->get_raw_view_requested())
            continue;

        const ::camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();
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
        ui_descriptor.raw_window->setCd(&Holovibes::instance().get_cd());
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

    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_disable_raw_view();

    // Wait for the raw view to be disabled for notify
    while (pipe->get_disable_raw_view_requested())
        continue;
}

void set_p_accu(UserInterfaceDescriptor& ui_descriptor, bool is_p_accu, uint p_value)
{
    Holovibes::instance().get_cd().p_accu_enabled = is_p_accu;
    Holovibes::instance().get_cd().p_acc_level = p_value;
    pipe_refresh(ui_descriptor);
}

void set_x_accu(UserInterfaceDescriptor& ui_descriptor, bool is_x_accu, uint x_value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().x_accu_enabled = is_x_accu;
    Holovibes::instance().get_cd().x_acc_level = x_value;
    pipe_refresh(ui_descriptor);
}

void set_y_accu(UserInterfaceDescriptor& ui_descriptor, bool is_y_accu, uint y_value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().y_accu_enabled = is_y_accu;
    Holovibes::instance().get_cd().y_acc_level = y_value;
    pipe_refresh(ui_descriptor);
}

void set_x_y(UserInterfaceDescriptor& ui_descriptor, const camera::FrameDescriptor& frame_descriptor, uint x, uint y)
{
    LOG_INFO;

    if (x < frame_descriptor.width)
        Holovibes::instance().get_cd().x_cuts = x;

    if (y < frame_descriptor.height)
        Holovibes::instance().get_cd().y_cuts = y;
}

void set_q(UserInterfaceDescriptor& ui_descriptor, int value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().q_index = value;
}

void set_q_accu(UserInterfaceDescriptor& ui_descriptor, bool is_q_accu, uint q_value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().q_acc_enabled = is_q_accu;
    Holovibes::instance().get_cd().q_acc_level = q_value;
    pipe_refresh(ui_descriptor);
}

const bool set_p(UserInterfaceDescriptor& ui_descriptor, int value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    if (value < static_cast<int>(Holovibes::instance().get_cd().time_transformation_size))
    {
        Holovibes::instance().get_cd().pindex = value;
        pipe_refresh(ui_descriptor);
        return true;
    }
    else
        LOG_ERROR << "p param has to be between 1 and #img";
    return false;
}

void set_composite_intervals(UserInterfaceDescriptor& ui_descriptor, uint composite_p_red, uint composite_p_blue)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_red = composite_p_red;
    Holovibes::instance().get_cd().composite_p_blue = composite_p_blue;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_h_min(UserInterfaceDescriptor& ui_descriptor, uint composite_p_min_h)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_min_h = composite_p_min_h;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_h_max(UserInterfaceDescriptor& ui_descriptor, uint composite_p_max_h)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_max_h = composite_p_max_h;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_s_min(UserInterfaceDescriptor& ui_descriptor, uint composite_p_min_s)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_min_s = composite_p_min_s;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_s_max(UserInterfaceDescriptor& ui_descriptor, uint composite_p_max_s)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_max_s = composite_p_max_s;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_v_min(UserInterfaceDescriptor& ui_descriptor, uint composite_p_min_v)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_min_v = composite_p_min_v;
    pipe_refresh(ui_descriptor);
}

void set_composite_intervals_hsv_v_max(UserInterfaceDescriptor& ui_descriptor, uint composite_p_max_v)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_max_v = composite_p_max_v;
    pipe_refresh(ui_descriptor);
}

void set_composite_weights(UserInterfaceDescriptor& ui_descriptor, uint weight_r, uint weight_g, uint weight_b)
{
    LOG_INFO;
    Holovibes::instance().get_cd().weight_r = weight_r;
    Holovibes::instance().get_cd().weight_g = weight_g;
    Holovibes::instance().get_cd().weight_b = weight_b;
    pipe_refresh(ui_descriptor);
}

void set_composite_auto_weights(::holovibes::gui::MainWindow& mainwindow,
                                UserInterfaceDescriptor& ui_descriptor,
                                bool value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_auto_weights_ = value;
    mainwindow.set_auto_contrast();
}

void select_composite_rgb(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    Holovibes::instance().get_cd().composite_kind = CompositeKind::RGB;
}

void select_composite_hsv(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    Holovibes::instance().get_cd().composite_kind = CompositeKind::HSV;
}

void actualize_frequency_channel_s(UserInterfaceDescriptor& ui_descriptor, bool composite_p_activated_s)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_activated_s = composite_p_activated_s;
}

void actualize_frequency_channel_v(UserInterfaceDescriptor& ui_descriptor, bool composite_p_activated_v)
{
    LOG_INFO;
    Holovibes::instance().get_cd().composite_p_activated_v = composite_p_activated_v;
}

void actualize_selection_h_gaussian_blur(UserInterfaceDescriptor& ui_descriptor, bool h_blur_activated)
{
    LOG_INFO;
    Holovibes::instance().get_cd().h_blur_activated = h_blur_activated;
}

void actualize_kernel_size_blur(UserInterfaceDescriptor& ui_descriptor, uint h_blur_kernel_size)
{
    LOG_INFO;
    Holovibes::instance().get_cd().h_blur_kernel_size = h_blur_kernel_size;
}

bool slide_update_threshold(const int slider_value,
                            std::atomic<float>& receiver,
                            std::atomic<float>& bound_to_update,
                            const std::atomic<float>& lower_bound,
                            const std::atomic<float>& upper_bound)
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

bool increment_p(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    if (Holovibes::instance().get_cd().pindex < Holovibes::instance().get_cd().time_transformation_size)
    {
        Holovibes::instance().get_cd().pindex++;
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

    if (Holovibes::instance().get_cd().pindex > 0)
    {
        Holovibes::instance().get_cd().pindex--;
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

    Holovibes::instance().get_cd().lambda = static_cast<float>(value) * 1.0e-9f;
    pipe_refresh(ui_descriptor);
    return true;
}

bool set_z(UserInterfaceDescriptor& ui_descriptor, const double value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    Holovibes::instance().get_cd().zdistance = static_cast<float>(value);
    pipe_refresh(ui_descriptor);
    return true;
}

bool increment_z(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    set_z(ui_descriptor, Holovibes::instance().get_cd().zdistance + ui_descriptor.z_step_);
    return true;
}

bool decrement_z(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    set_z(ui_descriptor, Holovibes::instance().get_cd().zdistance - ui_descriptor.z_step_);
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
        Holovibes::instance().get_cd().time_transformation = TimeTransformation::STFT;
    else if (value == "PCA")
        Holovibes::instance().get_cd().time_transformation = TimeTransformation::PCA;
    else if (value == "None")
        Holovibes::instance().get_cd().time_transformation = TimeTransformation::NONE;
    else if (value == "SSA_STFT")
        Holovibes::instance().get_cd().time_transformation = TimeTransformation::SSA_STFT;

    mainwindow.set_holographic_mode();
    return true;
}

void adapt_time_transformation_stride_to_batch_size(UserInterfaceDescriptor& ui_descriptor)
{
    if (Holovibes::instance().get_cd().time_transformation_stride < Holovibes::instance().get_cd().batch_size)
        Holovibes::instance().get_cd().time_transformation_stride = Holovibes::instance().get_cd().batch_size.load();
    // Go to lower multiple
    if (Holovibes::instance().get_cd().time_transformation_stride % Holovibes::instance().get_cd().batch_size != 0)
        Holovibes::instance().get_cd().time_transformation_stride -=
            Holovibes::instance().get_cd().time_transformation_stride % Holovibes::instance().get_cd().batch_size;
}

bool set_unwrapping_2d(UserInterfaceDescriptor& ui_descriptor, const bool value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    Holovibes::instance().get_compute_pipe()->request_unwrapping_2d(value);
    pipe_refresh(ui_descriptor);
    return true;
}

bool set_accumulation(UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    Holovibes::instance().get_cd().set_accumulation(Holovibes::instance().get_cd().current_window, value);
    pipe_refresh(ui_descriptor);
    return true;
}

bool set_accumulation_level(UserInterfaceDescriptor& ui_descriptor, int value)
{
    LOG_INFO;
    if (is_raw_mode(ui_descriptor))
        return false;

    Holovibes::instance().get_cd().set_accumulation_level(Holovibes::instance().get_cd().current_window, value);
    pipe_refresh(ui_descriptor);
    return true;
}

void set_composite_area(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    ui_descriptor.mainDisplay->getOverlayManager().create_overlay<::holovibes::gui::CompositeArea>();
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

void close_critical_compute(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (Holovibes::instance().get_cd().convolution_enabled)
        unset_convolution_mode(ui_descriptor);

    if (Holovibes::instance().get_cd().time_transformation_cuts_enabled)
        cancel_time_transformation_cuts(ui_descriptor, []() { return; });

    Holovibes::instance().stop_compute();
}

#pragma endregion

#pragma region Texture

void rotateTexture(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    const WindowKind curWin = Holovibes::instance().get_cd().current_window;

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

    const WindowKind curWin = Holovibes::instance().get_cd().current_window;

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

#pragma endregion

#pragma region Contrast - Log

bool set_contrast_mode(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    mainwindow.change_window();
    Holovibes::instance().get_cd().contrast_enabled = value;
    Holovibes::instance().get_cd().contrast_auto_refresh = true;
    pipe_refresh(ui_descriptor);
    return true;
}

void set_auto_contrast_cuts(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    if (!Holovibes::instance().get_cd().time_transformation_cuts_enabled)
    {
        return;
    }

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
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

bool set_contrast_min(UserInterfaceDescriptor& ui_descriptor, const double value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    if (Holovibes::instance().get_cd().contrast_enabled)
    {
        // Get the minimum contrast value rounded for the comparison
        const float old_val =
            Holovibes::instance().get_cd().get_truncate_contrast_min(Holovibes::instance().get_cd().current_window);
        // Floating number issue: cast to float for the comparison
        const float val = value;
        if (old_val != val)
        {
            Holovibes::instance().get_cd().set_contrast_min(Holovibes::instance().get_cd().current_window, value);
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

    if (Holovibes::instance().get_cd().contrast_enabled)
    {
        // Get the maximum contrast value rounded for the comparison
        const float old_val =
            Holovibes::instance().get_cd().get_truncate_contrast_max(Holovibes::instance().get_cd().current_window);
        // Floating number issue: cast to float for the comparison
        const float val = value;
        if (old_val != val)
        {
            Holovibes::instance().get_cd().set_contrast_max(Holovibes::instance().get_cd().current_window, value);
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

    if (Holovibes::instance().get_cd().contrast_enabled)
    {
        Holovibes::instance().get_cd().contrast_invert = value;
        pipe_refresh(ui_descriptor);
        return true;
    }

    return false;
}

void set_auto_refresh_contrast(UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().contrast_auto_refresh = value;
    pipe_refresh(ui_descriptor);
}

bool set_log_scale(UserInterfaceDescriptor& ui_descriptor, const bool value)
{
    LOG_INFO;

    if (is_raw_mode(ui_descriptor))
        return false;

    Holovibes::instance().get_cd().set_log_scale_slice_enabled(Holovibes::instance().get_cd().current_window, value);
    if (value && Holovibes::instance().get_cd().contrast_enabled)
        set_auto_contrast(ui_descriptor);

    pipe_refresh(ui_descriptor);
    return true;
}

#pragma endregion

#pragma region Convolution

bool update_convo_kernel(UserInterfaceDescriptor& ui_descriptor, const std::string& value)
{
    LOG_INFO;

    if (Holovibes::instance().get_cd().convolution_enabled)
    {
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
        return true;
    }

    return false;
}

void set_convolution_mode(UserInterfaceDescriptor& ui_descriptor, std::string& str)
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

void unset_convolution_mode(UserInterfaceDescriptor& ui_descriptor)
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

void set_divide_convolution_mode(UserInterfaceDescriptor& ui_descriptor, const bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().divide_convolution_enabled = value;

    pipe_refresh(ui_descriptor);
}

#pragma endregion

#pragma region Reticle

void display_reticle(UserInterfaceDescriptor& ui_descriptor, bool value)
{
    LOG_INFO;

    Holovibes::instance().get_cd().reticle_enabled = value;
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

    Holovibes::instance().get_cd().reticle_scale = value;
    pipe_refresh(ui_descriptor);
    return true;
}

#pragma endregion

#pragma region Chart

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
    if (Holovibes::instance().get_cd().chart_display_enabled)
        return;

    auto pipe = Holovibes::instance().get_compute_pipe();
    pipe->request_display_chart();

    // Wait for the chart display to be enabled for notify
    while (pipe->get_chart_display_requested())
        continue;

    ui_descriptor.plot_window_ = std::make_unique<::holovibes::gui::PlotWindow>(
        *Holovibes::instance().get_compute_pipe()->get_chart_display_queue(),
        ui_descriptor.auto_scale_point_threshold_,
        "Chart");
}

void stop_chart_display(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    if (!Holovibes::instance().get_cd().chart_display_enabled)
        return;

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

    ui_descriptor.plot_window_.reset(nullptr);
}

#pragma endregion

#pragma region Record

// Check that value is higher or equal than 0
void set_record_frame_step(UserInterfaceDescriptor& ui_descriptor, int value)
{
    ui_descriptor.record_frame_step_ = value;
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
        Holovibes::instance().start_batch_gpib(batch_input_path,
                                               output_path,
                                               nb_frames_to_record.value(),
                                               ui_descriptor.record_mode_,
                                               callback);
    }
    else
    {
        if (ui_descriptor.record_mode_ == RecordMode::CHART)
        {
            Holovibes::instance().start_chart_record(output_path, nb_frames_to_record.value(), callback);
        }
        else if (ui_descriptor.record_mode_ == RecordMode::HOLOGRAM)
        {
            Holovibes::instance().start_frame_record(output_path, nb_frames_to_record, false, 0, callback);
        }
        else if (ui_descriptor.record_mode_ == RecordMode::RAW)
        {
            Holovibes::instance().start_frame_record(output_path, nb_frames_to_record, true, 0, callback);
        }
    }
}

void stop_record(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;
    Holovibes::instance().stop_batch_gpib();

    if (ui_descriptor.record_mode_ == RecordMode::CHART)
        Holovibes::instance().stop_chart_record();
    else if (ui_descriptor.record_mode_ == RecordMode::HOLOGRAM || ui_descriptor.record_mode_ == RecordMode::RAW)
        Holovibes::instance().stop_frame_record();
}

void record_finished(UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    ui_descriptor.is_recording_ = false;
}

#pragma endregion

#pragma region Import

void import_stop(::holovibes::gui::MainWindow& mainwindow, UserInterfaceDescriptor& ui_descriptor)
{
    LOG_INFO;

    Holovibes::instance().stop_all_worker_controller();
    Holovibes::instance().start_information_display(false);

    close_critical_compute(ui_descriptor);

    // FIXME: import_stop() and camera_none() call same methods
    // FIXME: camera_none() weird call because we are dealing with imported file
    mainwindow.camera_none();

    Holovibes::instance().get_cd().is_computation_stopped = true;
}

bool import_start(::holovibes::gui::MainWindow& mainwindow,
                  UserInterfaceDescriptor& ui_descriptor,
                  std::string& file_path,
                  unsigned int fps,
                  size_t first_frame,
                  bool load_file_in_gpu,
                  size_t last_frame)
{
    LOG_INFO;

    if (!Holovibes::instance().get_cd().is_computation_stopped)
        // if computation is running
        import_stop(mainwindow, ui_descriptor);

    Holovibes::instance().get_cd().is_computation_stopped = false;
    // Gather all the usefull data from the ui import panel
    return init_holovibes_import_mode(ui_descriptor, file_path, fps, first_frame, load_file_in_gpu, last_frame);
}

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

#pragma endregion

} // namespace holovibes::api
