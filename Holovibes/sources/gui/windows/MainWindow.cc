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
    , holovibes(holovibes)
    , cd_(holovibes.get_cd())
    , ui_(new Ui::MainWindow)
{
    ui_->setupUi(this);
    panels_ = {ui_->ImageRenderingPanel,
               ui_->ViewPanel,
               ui_->CompositePanel,
               ui_->ImportPanel,
               ui_->ExportPanel,
               ui_->InfoPanel};

    qRegisterMetaType<std::function<void()>>();
    connect(this,
            SIGNAL(synchronize_thread_signal(std::function<void()>)),
            this,
            SLOT(synchronize_thread(std::function<void()>)));

    setWindowIcon(QIcon("Holovibes.ico"));

    auto display_info_text_fun = [=](const std::string& text) {
        synchronize_thread([=]() { ui_->InfoPanel->set_text(text.c_str()); });
    };
    Holovibes::instance().get_info_container().set_display_info_text_function(display_info_text_fun);

    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();

    // need the correct dimensions of main windows
    move(QPoint((screen_width - 800) / 2, (screen_height - 500) / 2));

    // Set default files
    std::filesystem::path holovibesdocuments_path = get_user_documents_path() / "Holovibes";
    std::filesystem::create_directory(holovibesdocuments_path);

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

    set_night();

    // Keyboard shortcuts
    QComboBox* window_cbox = ui_->WindowSelectionComboBox;
    connect(window_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(change_window()));

    // Display default values
    cd_.set_compute_mode(Computation::Raw);
    notify();
    setFocusPolicy(Qt::StrongFocus);

    // spinBox allow ',' and '.' as decimal point
    spinBoxDecimalPointReplacement(ui_->WaveLengthDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui_->ZDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui_->ContrastMaxDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui_->ContrastMinDoubleSpinBox);

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
        ui_->KernelQuickSelectComboBox->addItems(QStringList::fromVector(files));
    }

    // Initialize all panels
    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->init();

    Holovibes::instance().start_information_display(false);
}

MainWindow::~MainWindow()
{
    close_windows();
    close_critical_compute();
    camera_none();
    remove_infos();

    Holovibes::instance().stop_all_worker_controller();

    delete ui_;
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
        ui_->CompositePanel->hide();
        ui_->ImageRenderingPanel->setEnabled(false);
        ui_->ViewPanel->setEnabled(false);
        ui_->ExportPanel->setEnabled(false);
        layout_toggled();
        return;
    }

    if (is_enabled_camera_)
    {
        ui_->ImageRenderingPanel->setEnabled(true);
        ui_->ViewPanel->setEnabled(cd_.compute_mode == Computation::Hologram);
        ui_->ExportPanel->setEnabled(true);
    }

    ui_->CompositePanel->setHidden(is_raw_mode() || (cd_.img_type != ImgType::Composite));
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
                                                    ui_->ImportPanel->get_file_input_directory().c_str(),
                                                    tr("All files (*.ini);; Ini files (*.ini)"));

    reload_ini(filename);
}

void MainWindow::reload_ini() { reload_ini(""); }

void MainWindow::reload_ini(QString filename)
{
    ui_->ImportPanel->import_stop();
    try
    {
        load_ini(filename.isEmpty() ? ::holovibes::ini::get_global_ini_path() : filename.toStdString());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        LOG_INFO << e.what() << std::endl;
    }

    auto import_type = ui_->ImportPanel->get_import_type();
    if (import_type == ImportPanel::ImportType::File)
        ui_->ImportPanel->import_start();
    else if (import_type == ImportPanel::ImportType::Camera)
    {
        change_camera(kCamera);
    }
    notify();
}

void MainWindow::load_ini(const std::string& path)
{
    boost::property_tree::ptree ptree;
    boost::property_tree::ini_parser::read_ini(path, ptree);

    if (!ptree.empty())
    {
        // Load general compute data
        ini::load_ini(ptree, cd_);

        last_img_type_ = cd_.img_type == ImgType::Composite ? "Composite image" : last_img_type_;

        ui_->ViewModeComboBox->setCurrentIndex(static_cast<int>(cd_.img_type.load()));

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
    Config& config = global::global_config;

    // Save general compute data
    ini::save_ini(ptree, cd_);

    ptree.put<int>("image_rendering.camera", static_cast<int>(kCamera));

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
        ui_->ImageRenderingPanel->set_convolution_mode(false);

    if (cd_.time_transformation_cuts_enabled)
        ui_->ViewPanel->cancel_time_transformation_cuts();

    holovibes.stop_compute();
}

void MainWindow::camera_none()
{
    close_windows();
    close_critical_compute();
    if (!is_raw_mode())
        holovibes.stop_compute();
    holovibes.stop_frame_read();
    remove_infos();

    // Make camera's settings menu unaccessible
    ui_->actionSettings->setEnabled(false);
    is_enabled_camera_ = false;

    cd_.set_computation_stopped(true);
    notify();
}

void MainWindow::remove_infos() { Holovibes::instance().get_info_container().clear(); }

void MainWindow::close_windows()
{
    ui_->ViewPanel->sliceXZ.reset(nullptr);
    ui_->ViewPanel->sliceYZ.reset(nullptr);

    ui_->ExportPanel->plot_window.reset(nullptr);
    mainDisplay.reset(nullptr);

    ui_->ViewPanel->lens_window.reset(nullptr);
    ui_->ImageRenderingPanel->filter2d_window.reset(nullptr);

    /* Raw view & recording */
    ui_->ViewPanel->raw_window.reset(nullptr);

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
        holovibes.stop_compute();
    holovibes.stop_frame_read();
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
    holovibes.reload_streams();

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
                holovibes.stop_compute();
            holovibes.stop_frame_read();

            set_camera_timeout();

            ui_->ImageRenderingPanel->set_computation_mode();

            holovibes.start_camera_frame_read(c);
            is_enabled_camera_ = true;
            ui_->ImageRenderingPanel->set_image_mode(nullptr);
            ui_->ImportPanel->set_import_type(ImportPanel::ImportType::Camera);
            kCamera = c;

            // Make camera's settings menu accessible
            QAction* settings = ui_->actionSettings;
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
    open_file(std::filesystem::current_path().generic_string() + "/" + holovibes.get_camera_ini_path());
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
        holovibes.start_compute();
        holovibes.get_compute_pipe()->register_observer(*this);
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot create Pipe: " << e.what();
    }
}

void MainWindow::createHoloWindow()
{
    QPoint pos(0, 0);
    const FrameDescriptor& fd = holovibes.get_gpu_input_queue()->get_fd();
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
                                         holovibes.get_gpu_output_queue().get(),
                                         holovibes.get_compute_pipe(),
                                         ui_->ViewPanel->sliceXZ,
                                         ui_->ViewPanel->sliceYZ,
                                         this));
        mainDisplay->set_is_resize(false);
        mainDisplay->setTitle(QString("XY view"));
        mainDisplay->setCd(&cd_);
        mainDisplay->resetTransform();
        mainDisplay->setAngle(ui_->ViewPanel->displayAngle);
        mainDisplay->setFlip(ui_->ViewPanel->displayFlip);
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
    cd_.img_type = static_cast<ImgType>(ui_->ViewModeComboBox->currentIndex());
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

            ui_->PRedSpinBox_Composite->setValue(min_val_composite);
            ui_->SpinBox_hue_freq_min->setValue(min_val_composite);
            ui_->SpinBox_saturation_freq_min->setValue(min_val_composite);
            ui_->SpinBox_value_freq_min->setValue(min_val_composite);

            ui_->PBlueSpinBox_Composite->setValue(max_val_composite);
            ui_->SpinBox_hue_freq_max->setValue(max_val_composite);
            ui_->SpinBox_saturation_freq_max->setValue(max_val_composite);
            ui_->SpinBox_value_freq_max->setValue(max_val_composite);
        }
    }
    last_img_type_ = value;

    auto pipe = dynamic_cast<Pipe*>(holovibes.get_compute_pipe().get());

    pipe->insert_fn_end_vect([=]() {
        cd_.set_img_type(static_cast<ImgType>(ui_->ViewModeComboBox->currentIndex()));
        notify();
        layout_toggled();
    });
    pipe_refresh();

    // Force XYview autocontrast
    pipe->autocontrast_end_pipe(WindowKind::XYview);
    // Force cuts views autocontrast if needed
    if (cd_.time_transformation_cuts_enabled)
        ui_->ViewPanel->set_auto_contrast_cuts();
}

bool MainWindow::is_raw_mode() { return cd_.compute_mode == Computation::Raw; }
#pragma endregion
/* ------------ */
#pragma region Computation
void MainWindow::change_window()
{
    QComboBox* window_cbox = ui_->WindowSelectionComboBox;

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
        holovibes.get_compute_pipe()->soft_request_refresh();
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

void MainWindow::start_import(QString filename)
{
    ui_->ImportPanel->import_file(filename);
    ui_->ImportPanel->import_start();
}

RawWindow* MainWindow::get_main_display() { return mainDisplay.get(); }

Ui::MainWindow* MainWindow::get_ui() { return ui_; }

ComputeDescriptor& MainWindow::get_cd() { return cd_; }
} // namespace gui
} // namespace holovibes
