#include <filesystem>
#include <algorithm>
#include <list>
#include <atomic>

#include <QAction>
#include <QDesktopServices>
#include <QFileDialog>
#include <QMessageBox>
#include <QRect>
#include <QScreen>
#include <QShortcut>
#include <QStyleFactory>

#include <boost/algorithm/string.hpp>
#include <boost/property_tree/ini_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include "ui_mainwindow.h"
#include "MainWindow.hh"
#include "pipe.hh"
#include "logger.hh"
#include "config.hh"
#include "ini_config.hh"
#include "tools.hh"
#include "input_frame_file_factory.hh"
#include "update_exception.hh"
#include "accumulation_exception.hh"

#define MIN_IMG_NB_TIME_TRANSFORMATION_CUTS 8

namespace holovibes
{
using camera::Endianness;
using camera::FrameDescriptor;
namespace gui
{
namespace
{
void spinBoxDecimalPointReplacement(QDoubleSpinBox* doubleSpinBox)
{
    class DoubleValidator : public QValidator
    {
        const QValidator* old;

      public:
        DoubleValidator(const QValidator* old_)
            : QValidator(const_cast<QValidator*>(old_))
            , old(old_)
        {
        }

        void fixup(QString& input) const
        {
            input.replace(".", QLocale().decimalPoint());
            input.replace(",", QLocale().decimalPoint());
            old->fixup(input);
        }
        QValidator::State validate(QString& input, int& pos) const
        {
            fixup(input);
            return old->validate(input, pos);
        }
    };
    QLineEdit* lineEdit = doubleSpinBox->findChild<QLineEdit*>();
    lineEdit->setValidator(new DoubleValidator(lineEdit->validator()));
}
} // namespace
#pragma region Constructor - Destructor
MainWindow::MainWindow(Holovibes& holovibes, QWidget* parent)
    : QMainWindow(parent)
    , holovibes_(holovibes)
    , cd_(holovibes_.get_cd())
    , ui(new Ui::MainWindow)
{
    ui->setupUi(this);
    panels_ =
        {ui->ImageRenderingPanel, ui->ViewPanel, ui->CompositePanel, ui->ImportPanel, ui->ExportPanel, ui->InfoPanel};

    qRegisterMetaType<std::function<void()>>();
    connect(this,
            SIGNAL(synchronize_thread_signal(std::function<void()>)),
            this,
            SLOT(synchronize_thread(std::function<void()>)));

    setWindowIcon(QIcon("Holovibes.ico"));

    auto display_info_text_fun = [=](const std::string& text) {
        synchronize_thread([=]() { ui->InfoPanel->set_text(text.c_str()); });
    };
    Holovibes::instance().get_info_container().set_display_info_text_function(display_info_text_fun);

    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();

    // need the correct dimensions of main windows
    move(QPoint((screen_width - 800) / 2, (screen_height - 500) / 2));

    // Set default files
    std::filesystem::path holovibes_documents_path = get_user_documents_path() / "Holovibes";
    std::filesystem::create_directory(holovibes_documents_path);

    try
    {
        load_ini(::holovibes::ini::get_global_ini_path());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        LOG_WARN << ::holovibes::ini::get_global_ini_path() << ": Configuration file not found. "
                 << "Initialization with default values.";
        save_ini(::holovibes::ini::get_global_ini_path());
    }

    ui->ImageRenderingPanel->set_z_step(z_step_);
    set_night();

    // Keyboard shortcuts
    z_up_shortcut_ = new QShortcut(QKeySequence("Up"), this);
    z_up_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(z_up_shortcut_, SIGNAL(activated()), ui->ImageRenderingPanel, SLOT(increment_z()));

    z_down_shortcut_ = new QShortcut(QKeySequence("Down"), this);
    z_down_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(z_down_shortcut_, SIGNAL(activated()), ui->ImageRenderingPanel, SLOT(decrement_z()));

    p_left_shortcut_ = new QShortcut(QKeySequence("Left"), this);
    p_left_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_left_shortcut_, SIGNAL(activated()), ui->ViewPanel, SLOT(decrement_p()));

    p_right_shortcut_ = new QShortcut(QKeySequence("Right"), this);
    p_right_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_right_shortcut_, SIGNAL(activated()), ui->ViewPanel, SLOT(increment_p()));

    QComboBox* window_cbox = ui->WindowSelectionComboBox;
    connect(window_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(change_window()));

    // Display default values
    cd_.set_compute_mode(Computation::Raw);
    notify();
    setFocusPolicy(Qt::StrongFocus);

    // spinBox allow ',' and '.' as decimal point
    spinBoxDecimalPointReplacement(ui->WaveLengthDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui->ZDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui->ContrastMaxDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui->ContrastMinDoubleSpinBox);

    // Fill the quick kernel combo box with files from convolution_kernels
    // directory
    std::filesystem::path convo_matrix_path(get_exe_dir());
    convo_matrix_path = convo_matrix_path / "convolution_kernels";
    if (std::filesystem::exists(convo_matrix_path))
    {
        QVector<QString> files;
        for (const auto& file : std::filesystem::directory_iterator(convo_matrix_path))
        {
            files.push_back(QString(file.path().filename().string().c_str()));
        }
        std::sort(files.begin(), files.end(), [&](const auto& a, const auto& b) { return a < b; });
        ui->KernelQuickSelectComboBox->addItems(QStringList::fromVector(files));
    }

    // Initialize all panels
    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->init();

    Holovibes::instance().start_information_display(false);
}

MainWindow::~MainWindow()
{
    delete z_up_shortcut_;
    delete z_down_shortcut_;
    delete p_left_shortcut_;
    delete p_right_shortcut_;

    close_windows();
    close_critical_compute();
    camera_none();
    remove_infos();

    Holovibes::instance().stop_all_worker_controller();

    delete ui;
}

#pragma endregion
/* ------------ */
#pragma region Notify
void MainWindow::synchronize_thread(std::function<void()> f)
{
    // We can't update gui values from a different thread
    // so we pass it to the right one using a signal
    // (This whole notify thing needs to be cleaned up / removed)
    if (QThread::currentThread() != this->thread())
        emit synchronize_thread_signal(f);
    else
        f();
}

void MainWindow::notify()
{
    synchronize_thread([this]() { on_notify(); });
}

void MainWindow::on_notify()
{
    // Notify all panels
    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->on_notify();

    // Tabs
    if (cd_.is_computation_stopped)
    {
        ui->CompositePanel->hide();
        ui->ImageRenderingPanel->setEnabled(false);
        ui->ViewPanel->setEnabled(false);
        ui->ExportPanel->setEnabled(false);
        layout_toggled();
        return;
    }

    if (is_enabled_camera_)
    {
        ui->ImageRenderingPanel->setEnabled(true);
        ui->ViewPanel->setEnabled(cd_.compute_mode == Computation::Hologram);
        ui->ExportPanel->setEnabled(true);
    }

    ui->CompositePanel->setHidden(is_raw_mode() || (cd_.img_type != ImgType::Composite));
}

void MainWindow::notify_error(const std::exception& e)
{
    const CustomException* err_ptr = dynamic_cast<const CustomException*>(&e);
    if (err_ptr)
    {
        const UpdateException* err_update_ptr = dynamic_cast<const UpdateException*>(err_ptr);
        if (err_update_ptr)
        {
            auto lambda = [this] {
                // notify will be in close_critical_compute
                cd_.handle_update_exception();
                close_windows();
                close_critical_compute();
                LOG_ERROR << "GPU computing error occured.";
                notify();
            };
            synchronize_thread(lambda);
        }

        auto lambda = [this, accu = (dynamic_cast<const AccumulationException*>(err_ptr) != nullptr)] {
            if (accu)
            {
                cd_.handle_accumulation_exception();
            }
            close_critical_compute();

            LOG_ERROR << "GPU computing error occured.";
            notify();
        };
        synchronize_thread(lambda);
    }
    else
    {
        LOG_ERROR << "Unknown error occured.";
    }
}

void MainWindow::layout_toggled()
{

    synchronize_thread([=]() {
        // Resizing to original size, then adjust it to fit the groupboxes
        resize(baseSize());
        adjustSize();
    });
}

void MainWindow::credits()
{
    std::string msg = "Holovibes v" + std::string(__HOLOVIBES_VERSION__) +
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

    // Creation on the fly of the message box to display
    QMessageBox msg_box;
    msg_box.setText(QString::fromUtf8(msg.c_str()));
    msg_box.setIcon(QMessageBox::Information);
    msg_box.exec();
}

void MainWindow::documentation()
{
    QDesktopServices::openUrl(QUrl("https://ftp.espci.fr/incoming/Atlan/holovibes/manual/"));
}

#pragma endregion
/* ------------ */
#pragma region Ini

void MainWindow::configure_holovibes() { open_file(::holovibes::ini::get_global_ini_path()); }

void MainWindow::write_ini() { write_ini(""); }

void MainWindow::write_ini(QString filename)
{
    // Saves the current state of holovibes in holovibes.ini located in Holovibes.exe directory
    save_ini(filename.isEmpty() ? ::holovibes::ini::get_global_ini_path() : filename.toStdString());
    notify();
}

void MainWindow::browse_export_ini()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("All files (*.ini)"));
    write_ini(filename);
}

void MainWindow::browse_import_ini()
{
    QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("import .ini file"),
                                                    ui->ImportPanel->get_file_input_directory().c_str(),
                                                    tr("All files (*.ini);; Ini files (*.ini)"));

    reload_ini(filename);
}

void MainWindow::reload_ini() { reload_ini(""); }

void MainWindow::reload_ini(QString filename)
{
    ui->ImportPanel->import_stop();
    try
    {
        load_ini(filename.isEmpty() ? ::holovibes::ini::get_global_ini_path() : filename.toStdString());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        LOG_INFO << e.what() << std::endl;
    }

    auto import_type = ui->ImportPanel->get_import_type();
    if (import_type == ImportPanel::ImportType::File)
        ui->ImportPanel->import_start();
    else if (import_type == ImportPanel::ImportType::Camera)
    {
        change_camera(kCamera);
    }
    notify();
}

void MainWindow::load_ini(const std::string& path)
{
    boost::property_tree::ptree ptree;
    Panel* image_rendering_panel = ui->ImageRenderingPanel;
    Panel* view_panel = ui->ViewPanel;
    Panel* import_panel = ui->ImportPanel;
    Panel* info_panel = ui->InfoPanel;

    QAction* image_rendering_action = ui->actionImage_rendering;
    QAction* view_action = ui->actionView;
    QAction* import_export_action = ui->actionImportExport;
    QAction* info_action = ui->actionInfo;

    boost::property_tree::ini_parser::read_ini(path, ptree);

    if (!ptree.empty())
    {
        // Load general compute data
        ini::load_ini(ptree, cd_);

        image_rendering_action->setChecked(
            !ptree.get<bool>("image_rendering.hidden", image_rendering_panel->isHidden()));

        const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
        if (z_step > 0.0f)
            ui->ImageRenderingPanel->set_z_step(z_step);

        view_action->setChecked(!ptree.get<bool>("view.hidden", view_panel->isHidden()));

        last_img_type_ = cd_.img_type == ImgType::Composite ? "Composite image" : last_img_type_;

        ui->ViewModeComboBox->setCurrentIndex(static_cast<int>(cd_.img_type.load()));

        displayAngle = ptree.get("view.mainWindow_rotate", displayAngle);
        xzAngle = ptree.get<float>("view.xCut_rotate", xzAngle);
        yzAngle = ptree.get<float>("view.yCut_rotate", yzAngle);
        displayFlip = ptree.get("view.mainWindow_flip", displayFlip);
        xzFlip = ptree.get("view.xCut_flip", xzFlip);
        yzFlip = ptree.get("view.yCut_flip", yzFlip);

        auto_scale_point_threshold_ =
            ptree.get<size_t>("chart.auto_scale_point_threshold", auto_scale_point_threshold_);

        import_export_action->setChecked(!ptree.get<bool>("import_export.hidden", import_panel->isHidden()));

        ui->ImportInputFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));

        info_action->setChecked(!ptree.get<bool>("info.hidden", info_panel->isHidden()));
        theme_index_ = ptree.get<int>("info.theme_type", theme_index_);

        window_max_size = ptree.get<uint>("display.main_window_max_size", 768);
        auxiliary_window_max_size = ptree.get<uint>("display.auxiliary_window_max_size", 512);

        for (auto it = panels_.begin(); it != panels_.end(); it++)
            (*it)->load_ini(ptree);

        notify();
    }
}

void MainWindow::save_ini(const std::string& path)
{
    boost::property_tree::ptree ptree;
    Panel* image_rendering_panel = ui->ImageRenderingPanel;
    Panel* view_panel = ui->ViewPanel;
    Frame* import_export_frame = ui->ImportExportFrame;
    Panel* info_panel = ui->InfoPanel;
    Config& config = global::global_config;

    // Save general compute data
    ini::save_ini(ptree, cd_);

    ptree.put<bool>("image_rendering.hidden", image_rendering_panel->isHidden());

    ptree.put<int>("image_rendering.camera", static_cast<int>(kCamera));

    ptree.put<double>("image_rendering.z_step", z_step_);

    ptree.put<bool>("view.hidden", view_panel->isHidden());

    ptree.put<float>("view.mainWindow_rotate", displayAngle);
    ptree.put<float>("view.xCut_rotate", xzAngle);
    ptree.put<float>("view.yCut_rotate", yzAngle);
    ptree.put<int>("view.mainWindow_flip", displayFlip);
    ptree.put<int>("view.xCut_flip", xzFlip);
    ptree.put<int>("view.yCut_flip", yzFlip);

    ptree.put<size_t>("chart.auto_scale_point_threshold", auto_scale_point_threshold_);

    ptree.put<bool>("import_export.hidden", import_export_frame->isHidden());

    ptree.put<bool>("info.hidden", info_panel->isHidden());
    ptree.put<ushort>("info.theme_type", theme_index_);

    ptree.put<uint>("display.main_window_max_size", window_max_size);
    ptree.put<uint>("display.auxiliary_window_max_size", auxiliary_window_max_size);

    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->save_ini(ptree);

    boost::property_tree::write_ini(path, ptree);

    LOG_INFO << "Configuration file holovibes.ini overwritten at " << path << std::endl;
}

void MainWindow::open_file(const std::string& path)
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString(path.c_str())));
}
#pragma endregion
/* ------------ */
#pragma region Close Compute
void MainWindow::close_critical_compute()
{
    if (cd_.convolution_enabled)
        ui->ImageRenderingPanel->set_convolution_mode(false);

    if (cd_.time_transformation_cuts_enabled)
        ui->ViewPanel->cancel_time_transformation_cuts();

    holovibes_.stop_compute();
}

void MainWindow::camera_none()
{
    close_windows();
    close_critical_compute();
    if (!is_raw_mode())
        holovibes_.stop_compute();
    holovibes_.stop_frame_read();
    remove_infos();

    // Make camera's settings menu unaccessible
    ui->actionSettings->setEnabled(false);
    is_enabled_camera_ = false;

    cd_.set_computation_stopped(true);
    notify();
}

void MainWindow::remove_infos() { Holovibes::instance().get_info_container().clear(); }

void MainWindow::close_windows()
{
    ui->ViewPanel->sliceXZ.reset(nullptr);
    ui->ViewPanel->sliceYZ.reset(nullptr);

    ui->ExportPanel->plot_window.reset(nullptr);
    mainDisplay.reset(nullptr);

    ui->ViewPanel->lens_window.reset(nullptr);
    ui->ImageRenderingPanel->filter2d_window.reset(nullptr);

    /* Raw view & recording */
    ui->ViewPanel->raw_window.reset(nullptr);

    // Disable windows and overlays
    cd_.reset_windows_display();
}

void MainWindow::reset()
{
    Config& config = global::global_config;
    int device = 0;

    close_critical_compute();
    camera_none();
    qApp->processEvents();

    if (!is_raw_mode())
        holovibes_.stop_compute();
    holovibes_.stop_frame_read();
    cd_.reset_gui();
    is_enabled_camera_ = false;

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
    close_windows();
    remove_infos();
    holovibes_.reload_streams();

    try
    {
        load_ini(::holovibes::ini::get_global_ini_path());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        LOG_WARN << ::holovibes::ini::get_global_ini_path()
                 << ": Config file not found. It will use the default values.";
    }
    notify();
}

void MainWindow::closeEvent(QCloseEvent*)
{
    close_windows();
    if (!cd_.is_computation_stopped)
        close_critical_compute();
    camera_none();
    remove_infos();
    save_ini(::holovibes::ini::get_global_ini_path());
}
#pragma endregion
/* ------------ */
#pragma region Cameras
void MainWindow::change_camera(CameraKind c)
{
    camera_none();

    if (c != CameraKind::NONE)
    {
        try
        {
            mainDisplay.reset(nullptr);
            if (!is_raw_mode())
                holovibes_.stop_compute();
            holovibes_.stop_frame_read();

            set_camera_timeout();

            ui->ImageRenderingPanel->set_computation_mode();

            holovibes_.start_camera_frame_read(c);
            is_enabled_camera_ = true;
            ui->ImageRenderingPanel->set_image_mode(nullptr);
            ui->ImportPanel->set_import_type(ImportPanel::ImportType::Camera);
            kCamera = c;

            // Make camera's settings menu accessible
            QAction* settings = ui->actionSettings;
            settings->setEnabled(true);

            cd_.set_computation_stopped(false);
            notify();
        }
        catch (const camera::CameraException& e)
        {
            LOG_ERROR << "[CAMERA] " << e.what();
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what();
        }
    }
}

void MainWindow::camera_ids() { change_camera(CameraKind::IDS); }

void MainWindow::camera_phantom() { change_camera(CameraKind::Phantom); }

void MainWindow::camera_bitflow_cyton() { change_camera(CameraKind::BitflowCyton); }

void MainWindow::camera_hamamatsu() { change_camera(CameraKind::Hamamatsu); }

void MainWindow::camera_adimec() { change_camera(CameraKind::Adimec); }

void MainWindow::camera_xiq() { change_camera(CameraKind::xiQ); }

void MainWindow::camera_xib() { change_camera(CameraKind::xiB); }

void MainWindow::configure_camera()
{
    open_file(std::filesystem::current_path().generic_string() + "/" + holovibes_.get_camera_ini_path());
}
#pragma endregion
/* ------------ */
#pragma region Image Mode
void MainWindow::init_image_mode(QPoint& position, QSize& size)
{
    if (mainDisplay)
    {
        position = mainDisplay->framePosition();
        size = mainDisplay->size();
        mainDisplay.reset(nullptr);
    }
}

void MainWindow::createPipe()
{
    try
    {
        holovibes_.start_compute();
        holovibes_.get_compute_pipe()->register_observer(*this);
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot create Pipe: " << e.what();
    }
}

void MainWindow::createHoloWindow()
{
    QPoint pos(0, 0);
    const FrameDescriptor& fd = holovibes_.get_gpu_input_queue()->get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, window_max_size);
    QSize size(width, height);
    init_image_mode(pos, size);
    /* ---------- */
    try
    {
        mainDisplay.reset(new HoloWindow(pos,
                                         size,
                                         holovibes_.get_gpu_output_queue().get(),
                                         holovibes_.get_compute_pipe(),
                                         ui->ViewPanel->sliceXZ,
                                         ui->ViewPanel->sliceYZ,
                                         this));
        mainDisplay->set_is_resize(false);
        mainDisplay->setTitle(QString("XY view"));
        mainDisplay->setCd(&cd_);
        mainDisplay->resetTransform();
        mainDisplay->setAngle(displayAngle);
        mainDisplay->setFlip(displayFlip);
        mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "createHoloWindow: " << e.what();
    }
}

void MainWindow::set_camera_timeout() { camera::FRAME_TIMEOUT = global::global_config.frame_timeout; }

void MainWindow::refreshViewMode()
{
    float old_scale = 1.f;
    glm::vec2 old_translation(0.f, 0.f);
    if (mainDisplay)
    {
        old_scale = mainDisplay->getScale();
        old_translation = mainDisplay->getTranslate();
    }
    close_windows();
    close_critical_compute();
    cd_.img_type = static_cast<ImgType>(ui->ViewModeComboBox->currentIndex());
    try
    {
        createPipe();
        createHoloWindow();
        mainDisplay->setScale(old_scale);
        mainDisplay->setTranslate(old_translation[0], old_translation[1]);
    }
    catch (const std::runtime_error& e)
    {
        mainDisplay.reset(nullptr);
        LOG_ERROR << "refreshViewMode: " << e.what();
    }
    notify();
    layout_toggled();
}

namespace
{
// Is there a change in window pixel depth (needs to be re-opened)
bool need_refresh(const QString& last_type, const QString& new_type)
{
    std::vector<QString> types_needing_refresh({"Composite image"});
    for (auto& type : types_needing_refresh)
        if ((last_type == type) != (new_type == type))
            return true;
    return false;
}
} // namespace
void MainWindow::set_view_image_type(const QString& value)
{
    if (is_raw_mode())
        return;

    if (need_refresh(last_img_type_, value))
    {
        refreshViewMode();
        if (cd_.img_type == ImgType::Composite)
        {
            const unsigned min_val_composite = cd_.time_transformation_size == 1 ? 0 : 1;
            const unsigned max_val_composite = cd_.time_transformation_size - 1;

            ui->PRedSpinBox_Composite->setValue(min_val_composite);
            ui->SpinBox_hue_freq_min->setValue(min_val_composite);
            ui->SpinBox_saturation_freq_min->setValue(min_val_composite);
            ui->SpinBox_value_freq_min->setValue(min_val_composite);

            ui->PBlueSpinBox_Composite->setValue(max_val_composite);
            ui->SpinBox_hue_freq_max->setValue(max_val_composite);
            ui->SpinBox_saturation_freq_max->setValue(max_val_composite);
            ui->SpinBox_value_freq_max->setValue(max_val_composite);
        }
    }
    last_img_type_ = value;

    auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());

    pipe->insert_fn_end_vect([=]() {
        cd_.set_img_type(static_cast<ImgType>(ui->ViewModeComboBox->currentIndex()));
        notify();
        layout_toggled();
    });
    pipe_refresh();

    // Force XYview autocontrast
    pipe->autocontrast_end_pipe(WindowKind::XYview);
    // Force cuts views autocontrast if needed
    if (cd_.time_transformation_cuts_enabled)
        ui->ViewPanel->set_auto_contrast_cuts();
}

bool MainWindow::is_raw_mode() { return cd_.compute_mode == Computation::Raw; }
#pragma endregion
/* ------------ */
#pragma region Computation
void MainWindow::change_window()
{
    QComboBox* window_cbox = ui->WindowSelectionComboBox;

    cd_.change_window(window_cbox->currentIndex());
    pipe_refresh();
    notify();
}

void MainWindow::pipe_refresh()
{
    if (is_raw_mode())
        return;

    try
    {
        holovibes_.get_compute_pipe()->soft_request_refresh();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what();
    }
}

#pragma endregion
/* ------------ */
#pragma region Themes
void MainWindow::set_night()
{
    // Dark mode style
    qApp->setStyle(QStyleFactory::create("Fusion"));

    QPalette darkPalette;
    darkPalette.setColor(QPalette::Window, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::WindowText, Qt::white);
    darkPalette.setColor(QPalette::Base, QColor(25, 25, 25));
    darkPalette.setColor(QPalette::AlternateBase, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ToolTipBase, Qt::white);
    darkPalette.setColor(QPalette::ToolTipText, Qt::white);
    darkPalette.setColor(QPalette::Text, Qt::white);
    darkPalette.setColor(QPalette::Button, QColor(53, 53, 53));
    darkPalette.setColor(QPalette::ButtonText, Qt::white);
    darkPalette.setColor(QPalette::BrightText, Qt::red);
    darkPalette.setColor(QPalette::Disabled, QPalette::Text, Qt::darkGray);
    darkPalette.setColor(QPalette::Disabled, QPalette::ButtonText, Qt::darkGray);
    darkPalette.setColor(QPalette::Disabled, QPalette::WindowText, Qt::darkGray);
    darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::HighlightedText, Qt::black);
    darkPalette.setColor(QPalette::Light, Qt::black);

    qApp->setPalette(darkPalette);
    theme_index_ = 1;
}

void MainWindow::set_classic()
{
    qApp->setPalette(this->style()->standardPalette());
    // Light mode style
    qApp->setStyle(QStyleFactory::create("WindowsVista"));
    qApp->setStyleSheet("");
    theme_index_ = 0;
}
#pragma endregion

RawWindow* MainWindow::get_main_display() { return mainDisplay.get(); }
} // namespace gui
} // namespace holovibes
