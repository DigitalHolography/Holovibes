#include <QAction>
#include <QDesktopServices>
#include <QFileDialog>
#include <QMessageBox>
#include <QRect>
#include <QScreen>
#include <QShortcut>
#include <QStyleFactory>

#include "MainWindow.hh"
#include "logger.hh"
#include "holovibes_config.hh"
#include "update_exception.hh"
#include "accumulation_exception.hh"
#include "gui_group_box.hh"
#include "tools.hh"
#include "logger.hh"
#include "camera_dll.hh"
#include "image_rendering_panel.hh"

#include "API.hh"
#include "GUI.hh"
#include "user_interface_descriptor.hh"

#include "view_struct.hh"

namespace holovibes
{
using camera::Endianness;
using camera::FrameDescriptor;
} // namespace holovibes

namespace holovibes::gui
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

MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
    , ui_(new Ui::MainWindow)
    , acquisition_started_subscriber_("acquisition_started",
                                      [this](bool success) { acquisition_finished_notification_received = false; })
    , acquisition_finished_subscriber_("acquisition_finished",
                                       [this](bool success)
                                       {
                                           if (acquisition_finished_notification_received)
                                               return;
                                           acquisition_finished_notification_received = true;
                                           ui_->InfoPanel->set_recordProgressBar_color(QColor(48, 143, 236),
                                                                                       "Saving: %v/%m");
                                           light_ui_->set_recordProgressBar_color(QColor(48, 143, 236), "Saving...");
                                       })
    , set_preset_subscriber_("set_preset_file_gpu", [this](bool success) { set_preset_file_on_gpu(); })
    , api_(API)
    , notify_subscriber_("notify", [this](bool success) { notify(); })
{
    disable_notify();

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

    setWindowIcon(QIcon(":/assets/icons/holovibes_logo.png"));

    ::holovibes::worker::InformationWorker::display_info_text_function_ = [=](const std::string& text)
    { synchronize_thread([=]() { ui_->InfoPanel->set_text(text.c_str()); }); };

    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();

    // need the correct dimensions of main windows
    move(QPoint((screen_width - 800) / 2, (screen_height - 500) / 2));

    // Set default files
    std::filesystem::path holovibes_documents_path = get_user_documents_path() / __APPNAME__;
    std::filesystem::create_directory(holovibes_documents_path);
    std::filesystem::create_directory(std::filesystem::path(__APPDATA_HOLOVIBES_FOLDER__));
    std::filesystem::create_directory(std::filesystem::path(__CONFIG_FOLDER__));

    // Fill the quick kernel combo box with files from convolution_kernels
    // directory
    std::filesystem::path convo_matrix_path(RELATIVE_PATH(__CONVOLUTION_KERNEL_FOLDER_PATH__));
    if (std::filesystem::exists(convo_matrix_path))
    {
        QVector<QString> files;
        files.push_back(QString(UID_CONVOLUTION_TYPE_DEFAULT));
        for (const auto& file : std::filesystem::directory_iterator(convo_matrix_path))
            files.push_back(QString(file.path().filename().string().c_str()));
        std::sort(files.begin(), files.end(), [&](const auto& a, const auto& b) { return a < b; });
        ui_->KernelQuickSelectComboBox->addItems(QStringList::fromVector(files));
    }

    // Fill the input filter combo box with files from input_filters
    // directory
    std::filesystem::path input_filters_path(RELATIVE_PATH(__INPUT_FILTER_FOLDER_PATH__));
    if (std::filesystem::exists(input_filters_path))
    {
        QVector<QString> files;
        for (const auto& file : std::filesystem::directory_iterator(input_filters_path))
            files.push_back(QString(file.path().filename().string().c_str()));
        std::sort(files.begin(), files.end(), [&](const auto& a, const auto& b) { return a < b; });
        files.push_front(QString(UID_FILTER_TYPE_DEFAULT));
        ui_->InputFilterQuickSelectComboBox->addItems(QStringList::fromVector(files));
    }

    try
    {
        api_.settings.load_compute_settings(holovibes::settings::compute_settings_filepath);
        // Set values not set by notify
        ui_->BatchSizeSpinBox->setValue(api_.transform.get_batch_size());
    }
    catch (const std::exception&)
    {
        LOG_INFO("{}: Compute settings incorrect or file not found. Initialization with default values.",
                 ::holovibes::settings::compute_settings_filepath);
        api_.settings.save_compute_settings(holovibes::settings::compute_settings_filepath);
    }

    // Store the value because when the camera is initialised it is reset
    bool is_conv_enabled = api_.global_pp.get_convolution_enabled();

    // light ui
    light_ui_ = std::make_shared<LightUI>(nullptr, this);

    load_gui();

    if (api_.input.get_import_type() != ImportType::None)
        start();

    setFocusPolicy(Qt::StrongFocus);

    // spinBox allow ',' and '.' as decimal point
    spinBoxDecimalPointReplacement(ui_->LambdaSpinBox);
    spinBoxDecimalPointReplacement(ui_->ZDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui_->ContrastMaxDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui_->ContrastMinDoubleSpinBox);

    // Initialize all panels
    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->init();

    api_.information.start_information_display();

    ui_->ImageRenderingPanel->set_convolution_mode(is_conv_enabled);
    // Add the convolution after the initialisation of the panel
    // if the value is enabled in the compute settings.

    if (api_.window_pp.get_enabled(WindowKind::YZview) && api_.window_pp.get_enabled(WindowKind::XZview))
        ui_->ViewPanel->update_3d_cuts_view(true);

    init_tooltips();

    enable_notify();

    notify();
}

MainWindow::~MainWindow()
{
    ui_->menuSelect_preset->clear();

    gui::close_windows();
    api_.compute.close_critical_compute();
    api_.compute.stop_all_worker_controller();
    api_.input.set_camera_kind(CameraKind::NONE, false);

    delete ui_;
}

#pragma endregion

#pragma region Notify
void MainWindow::synchronize_thread(std::function<void()> f)
{
    // Ensure GUI updates are done on the main thread
    if (QThread::currentThread() != this->thread())
        emit synchronize_thread_signal(f);
    else
        f();
}

void MainWindow::enable_notify() { notify_enabled_ = true; }

void MainWindow::disable_notify() { notify_enabled_ = false; }

void MainWindow::notify()
{
    if (notify_enabled_)
        synchronize_thread([this]() { on_notify(); });
}

void MainWindow::on_notify()
{
    // Disable pipe refresh to avoid the numerous refreshes at the launch of the program
    api_.compute.disable_pipe_refresh();

    // Disable the notify for the same reason
    disable_notify();

    // Notify all panels
    for (auto& panel : panels_)
        panel->on_notify();

    enable_notify();

    api_.compute.enable_pipe_refresh();

    QSpinBox* fps_number_sb = this->findChild<QSpinBox*>("ImportInputFpsSpinBox");
    fps_number_sb->setValue(api_.input.get_input_fps());

    // Refresh the preset drop down menu
    ui_->menuSelect_preset->clear();
    std::filesystem::path preset_dir(RELATIVE_PATH(__PRESET_FOLDER_PATH__));
    if (!std::filesystem::exists(preset_dir))
        ui_->menuSelect_preset->addAction(new QAction(QString("Presets directory not found"), ui_->menuSelect_preset));
    else
    {
        QList<QAction*> actions;
        for (const auto& file : std::filesystem::directory_iterator(RELATIVE_PATH(__PRESET_FOLDER_PATH__)))
        {
            QAction* action = new QAction(QString(file.path().filename().string().c_str()), ui_->menuSelect_preset);
            connect(action, &QAction::triggered, this, [=] { set_preset(file); });
            actions.push_back(action);
        }
        if (actions.length() == 0)
            ui_->menuSelect_preset->addAction(new QAction(QString("No preset"), ui_->menuSelect_preset));
        else
            ui_->menuSelect_preset->addActions(actions);
    }

    // Tabs
    if (api_.compute.get_is_computation_stopped())
    {
        ui_->CompositePanel->hide();
        ui_->ImageRenderingPanel->setEnabled(false);
        ui_->ViewPanel->setEnabled(false);
        ui_->ExportPanel->setEnabled(false);
        light_ui_->pipeline_active(false);
        layout_toggled();
        return;
    }

    if (api_.input.get_import_type() != ImportType::None)
    {
        ui_->ImageRenderingPanel->setEnabled(true);
        ui_->ViewPanel->setEnabled(api_.compute.get_compute_mode() == Computation::Hologram);
        ui_->ExportPanel->setEnabled(true);
        light_ui_->pipeline_active(true);
    }

    ui_->CompositePanel->setHidden(api_.compute.get_compute_mode() == Computation::Raw ||
                                   (api_.compute.get_img_type() != ImgType::Composite));

    ui_->actionSettings->setEnabled(api_.input.get_camera_kind() != CameraKind::NONE);

    resize(baseSize());
    adjustSize();
}

void MainWindow::notify_error(const std::exception& e)
{
    const CustomException* err_ptr = dynamic_cast<const CustomException*>(&e);
    if (err_ptr)
    {
        const UpdateException* err_update_ptr = dynamic_cast<const UpdateException*>(err_ptr);
        if (err_update_ptr)
        {
            auto lambda = [&, this]
            {
                // notify will be in close_critical_compute
                api_.transform.set_time_transformation_size(1);
                api_.filter2d.enable_filter("");
                gui::close_windows();
                api_.compute.close_critical_compute();
                LOG_ERROR("GPU computing error occured. : {}", e.what());
                notify();
            };
            synchronize_thread(lambda);
        }

        auto lambda = [&, this, accu = (dynamic_cast<const AccumulationException*>(err_ptr) != nullptr)]
        {
            if (accu)
                api_.window_pp.set_accumulation_level(1, WindowKind::XYview);
            api_.compute.close_critical_compute();

            LOG_ERROR("GPU computing error occured. : {}", e.what());
            notify();
        };
        synchronize_thread(lambda);
    }
    else
    {
        LOG_ERROR("Unknown error occured. : {}", e.what());
    }
}

void MainWindow::layout_toggled()
{
    synchronize_thread(
        [=]()
        {
            // Resizing to original size, then adjust it to fit the groupboxes
            resize(baseSize());
            adjustSize();
        });
}

void MainWindow::credits()
{
    const std::vector<std::string> columns = api_.information.get_credits();

    // Create HTML for 3 columns inside a table
    QString columnarText = QString("Holovibes v" + QString::fromStdString(std::string(__HOLOVIBES_VERSION__)) +
                                   "\n\nDevelopers:\n\n"
                                   "<table width='100%%' cellspacing='20'>"
                                   "<tr>"
                                   "<td width='33%%' valign='top'>%1</td>"
                                   "<td width='33%%' valign='top'>%2</td>"
                                   "<td width='33%%' valign='top'>%3</td>"
                                   "</tr>"
                                   "</table>")
                               .arg(QString::fromStdString(columns[0]))
                               .arg(QString::fromStdString(columns[1]))
                               .arg(QString::fromStdString(columns[2]));

    // Creation on the fly of the message box to display
    QMessageBox msg_box;
    msg_box.setTextFormat(Qt::RichText); // Enable HTML formatting
    msg_box.setText(columnarText);       // Set the HTML-formatted text
    msg_box.setIcon(QMessageBox::Information);
    msg_box.exec();
}

void MainWindow::documentation()
{
    QDesktopServices::openUrl(QUrl(QString::fromStdString(api_.information.get_documentation_url())));
}

#pragma endregion

#pragma region Json

void MainWindow::write_compute_settings() { api_.settings.save_compute_settings(); }

void MainWindow::browse_export_ini()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("All files (*.json)"));
    api_.settings.save_compute_settings(filename.toStdString());
    // Reload to change drop down preset menu
    notify();
}

void MainWindow::reload_ini(const std::string& filename)
{
    ImportType it = api_.input.get_import_type();
    ui_->ImportPanel->import_stop();

    try
    {
        api_.settings.load_compute_settings(filename);
        // Set values not set by notify
        ui_->BatchSizeSpinBox->setValue(api_.transform.get_batch_size());
    }
    catch (const std::exception&)
    {
        LOG_INFO("{}: Compute settings incorrect or file not found. Initialization with default values.",
                 ::holovibes::settings::compute_settings_filepath);
        api_.settings.save_compute_settings(holovibes::settings::compute_settings_filepath);
    }

    if (it == ImportType::File)
        ui_->ImportPanel->import_start();
    else if (it == ImportType::Camera)
        change_camera(api_.input.get_camera_kind());
    else // if (it == ImportType::None)
        notify();
}

void MainWindow::browse_import_ini()
{
    QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("import .json file"),
                                                    UserInterfaceDescriptor::instance().file_input_directory_.c_str(),
                                                    tr("All files (*.json);; Json files (*.json)"));

    if (!filename.isEmpty())
        reload_ini(filename.toStdString());
}

void MainWindow::reload_ini() { reload_ini(::holovibes::settings::compute_settings_filepath); }

void set_module_visibility(QAction*& action, GroupBox*& groupbox, bool to_hide)
{
    action->setChecked(!to_hide);
    groupbox->setHidden(to_hide);
}

void MainWindow::load_gui()
{
    if (holovibes::settings::user_settings_filepath.empty())
        return;

    json j_us;

    try
    {
        std::ifstream ifs(settings::user_settings_filepath);
        j_us = json::parse(ifs);
    }
    catch (json::parse_error)
    {
        LOG_INFO("{} : User settings file not found. Initialization with default values.",
                 ::holovibes::settings::user_settings_filepath);
        save_gui();
    }

    set_theme(json_get_or_default(j_us, Theme::Dark, "display", "theme"));

    setBaseSize(json_get_or_default(j_us, 879, "main window", "width"),
                json_get_or_default(j_us, 470, "main window", "height"));
    resize(baseSize());
    move(json_get_or_default(j_us, 560, "main window", "x"), json_get_or_default(j_us, 290, "main window", "y"));

    window_max_size = json_get_or_default(j_us, window_max_size, "windows", "main window max size");
    auxiliary_window_max_size =
        json_get_or_default(j_us, auxiliary_window_max_size, "windows", "auxiliary window max size");

    light_ui_->set_window_size_position(json_get_or_default(j_us, 400, "light ui window", "width"),
                                        json_get_or_default(j_us, 115, "light ui window", "height"),
                                        json_get_or_default(j_us, 560, "light ui window", "x"),
                                        json_get_or_default(j_us, 290, "light ui window", "y"));

    api_.view.set_display_rate(json_get_or_default(j_us, api_.view.get_display_rate(), "display", "refresh rate"));
    api_.window_pp.set_raw_bitshift(
        json_get_or_default(j_us, api_.window_pp.get_raw_bitshift(), "file info", "raw bit shift"));

    ui_->ExportPanel->set_record_frame_step(
        json_get_or_default(j_us, ui_->ExportPanel->get_record_frame_step(), "gui settings", "record frame step"));

    UserInterfaceDescriptor::instance().auto_scale_point_threshold_ =
        json_get_or_default(j_us,
                            UserInterfaceDescriptor::instance().auto_scale_point_threshold_,
                            "chart",
                            "auto scale point threshold");
    UserInterfaceDescriptor::instance().output_filename_ =
        json_get_or_default(j_us,
                            UserInterfaceDescriptor::instance().output_filename_,
                            "files",
                            "default output filename");
    UserInterfaceDescriptor::instance().record_output_directory_ =
        json_get_or_default(j_us,
                            UserInterfaceDescriptor::instance().record_output_directory_,
                            "files",
                            "record output directory");
    UserInterfaceDescriptor::instance().file_input_directory_ =
        json_get_or_default(j_us,
                            UserInterfaceDescriptor::instance().file_input_directory_,
                            "files",
                            "file input directory");

    auto camera = json_get_or_default(j_us, CameraKind::NONE, "camera", "type");

    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->load_gui(j_us);

    bool is_camera = api_.input.set_camera_kind(camera);
}

void MainWindow::set_preset_file_on_gpu()
{
    std::filesystem::path dest = RELATIVE_PATH(__PRESET_FOLDER_PATH__ / "FILE_ON_GPU.json");
    api_.settings.import_buffer(dest.string());
    LOG_INFO("Preset loaded");
}

void MainWindow::save_gui()
{
    if (holovibes::settings::user_settings_filepath.empty())
        return;

    json j_us;

    auto path = holovibes::settings::user_settings_filepath;
    std::ifstream input_file(path);
    try
    {
        j_us = json::parse(input_file);
    }
    catch (const std::exception&)
    {
    }

    j_us["display"]["theme"] = theme_;

    j_us["windows"]["main window max size"] = window_max_size;
    j_us["windows"]["auxiliary window max size"] = auxiliary_window_max_size;

    j_us["main window"]["width"] = size().width();
    j_us["main window"]["height"] = size().height();
    j_us["main window"]["x"] = pos().x();
    j_us["main window"]["y"] = pos().y();

    j_us["light ui window"]["width"] = light_ui_->size().width();
    j_us["light ui window"]["height"] = light_ui_->size().height();
    j_us["light ui window"]["x"] = light_ui_->pos().x();
    j_us["light ui window"]["y"] = light_ui_->pos().y();

    j_us["display"]["refresh rate"] = api_.view.get_display_rate();
    j_us["file info"]["raw bit shift"] = api_.window_pp.get_raw_bitshift();
    j_us["gui settings"]["record frame step"] = ui_->ExportPanel->get_record_frame_step();
    j_us["chart"]["auto scale point threshold"] = UserInterfaceDescriptor::instance().auto_scale_point_threshold_;
    j_us["files"]["default output filename"] = UserInterfaceDescriptor::instance().output_filename_;
    j_us["files"]["record output directory"] = UserInterfaceDescriptor::instance().record_output_directory_;
    j_us["files"]["file input directory"] = UserInterfaceDescriptor::instance().file_input_directory_;

    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->save_gui(j_us);

    std::ofstream file(path);
    file << j_us.dump(1);

    LOG_INFO("user settings overwritten at {}", path);
}

#pragma endregion

#pragma region Close Compute

void MainWindow::closeEvent(QCloseEvent*)
{
    save_gui();
    if (save_cs)
        api_.settings.save_compute_settings();

    gui::close_windows();
    api_.input.set_camera_kind(CameraKind::NONE, false);
    Logger::flush();
}

#pragma endregion

#pragma region Cameras

void MainWindow::change_camera(CameraKind c)
{
    ui_->ImportPanel->import_stop();

    if (api_.input.set_camera_kind(c))
    {
        start();
        notify();
    }
}

void MainWindow::camera_none() { change_camera(CameraKind::NONE); }

void MainWindow::camera_ids() { change_camera(CameraKind::IDS); }

void MainWindow::camera_phantom() { change_camera(CameraKind::Phantom); }

void MainWindow::camera_bitflow_cyton() { change_camera(CameraKind::BitflowCyton); }

void MainWindow::camera_hamamatsu() { change_camera(CameraKind::Hamamatsu); }

void MainWindow::camera_adimec() { change_camera(CameraKind::Adimec); }

void MainWindow::camera_xiq() { change_camera(CameraKind::xiQ); }

void MainWindow::camera_xib() { change_camera(CameraKind::xiB); }

void MainWindow::camera_opencv() { change_camera(CameraKind::OpenCV); }

void MainWindow::camera_ametek_s991_coaxlink_qspf_plus() { change_camera(CameraKind::AmetekS991EuresysCoaxlinkQSFP); }

void MainWindow::camera_ametek_s711_coaxlink_qspf_plus() { change_camera(CameraKind::AmetekS711EuresysCoaxlinkQSFP); }

void MainWindow::auto_detection_phantom() { change_camera(CameraKind::AutoDetectionPhantom); }

void MainWindow::camera_euresys_egrabber() { change_camera(CameraKind::Ametek); }

void MainWindow::camera_alvium() { change_camera(CameraKind::Alvium); }

void MainWindow::configure_camera()
{
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString::fromStdString(api_.input.get_camera_ini_name())));
}

void open_file(const std::string& filename)
{
    if (filename.empty())
        return;

    QDesktopServices::openUrl(QUrl::fromLocalFile(
        QString::fromStdString((RELATIVE_PATH(__CAMERAS_CONFIG_FOLDER_PATH__ / filename)).string())));
}

void MainWindow::camera_ids_settings() { open_file("ids.ini"); }

void MainWindow::camera_phantom_settings() { open_file("ametek_s710_euresys_coaxlink_octo.ini"); }

void MainWindow::camera_bitflow_cyton_settings() { open_file("bitflow.ini"); }

void MainWindow::camera_hamamatsu_settings() { open_file("hamamatsu.ini"); }

void MainWindow::camera_adimec_settings() { open_file("adimec.ini"); }

void MainWindow::camera_xiq_settings() { open_file("xiq.ini"); }

void MainWindow::camera_xib_settings() { open_file("xib.ini"); }

void MainWindow::camera_opencv_settings() { open_file("opencv.ini"); }

void MainWindow::camera_ametek_s991_coaxlink_qspf_plus_settings()
{
    open_file("ametek_s991_euresys_coaxlink_qsfp+.ini");
}

void MainWindow::camera_ametek_s711_coaxlink_qspf_plus_settings()
{
    open_file("ametek_s711_euresys_coaxlink_qsfp+.ini");
}

void MainWindow::camera_euresys_egrabber_settings() { open_file("ametek_s710_euresys_coaxlink_octo.ini"); }

void MainWindow::camera_alvium_settings() { open_file("alvium.ini"); }

#pragma endregion

void MainWindow::change_window(int index)
{
    api_.view.change_window(static_cast<WindowKind>(index));

    notify();
}

void MainWindow::start_import(QString filename)
{
    ui_->ImportPanel->import_file(filename);
    start();
}

void MainWindow::start()
{
    gui::close_windows();
    if (api_.compute.start() == ApiCode::OK)
    {
        gui::create_window(api_.compute.get_compute_mode(), window_max_size);
        shift_screen();
    }
}

Ui::MainWindow* MainWindow::get_ui() { return ui_; }

#pragma endregion

#pragma region Advanced

void MainWindow::open_advanced_settings()
{
    if (UserInterfaceDescriptor::instance().advanced_settings_window_)
        return;

    gui::open_advanced_settings(this,
                                [=]()
                                {
                                    ImportType it = api_.input.get_import_type();

                                    if (it == ImportType::File)
                                        ui_->ImportPanel->import_start();
                                    else if (it == ImportType::Camera)
                                        change_camera(api_.input.get_camera_kind());
                                });
}

void MainWindow::reset_settings()
{
    std::string to_remove = holovibes::settings::compute_settings_filepath;

    std::stringstream tmp;
    tmp << "Reset settings and quit\n\nThis will remove the compute settings located in " << to_remove
        << " and Holovibe will close";

    QMessageBox msgBox;
    msgBox.setText(QString::fromUtf8(tmp.str().c_str()));
    msgBox.setStandardButtons(QMessageBox::Cancel | QMessageBox::Ok);
    msgBox.setIcon(QMessageBox::Warning);
    msgBox.setDefaultButton(QMessageBox::Cancel);

    int ret = msgBox.exec();
    switch (ret)
    {
    case QMessageBox::Cancel:
        break;
    case QMessageBox::Ok:
        if (std::remove(to_remove.c_str()) == 0)
        {
            save_cs = false;
            LOG_INFO("{} has been removed!", to_remove);
            LOG_INFO("Please, restart Holovibes!");
        }
        else
            LOG_WARN("Could not remove {}!", to_remove);

        close();
        break;
    }
}

#pragma endregion

#pragma region UI

void MainWindow::shift_screen()
{
    return;
    // we want to remain at the position indicated in the user_settings.json
    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();
    move(QPoint(210 + (screen_width - 800) / 2, 200 + (screen_height - 500) / 2));
}

void MainWindow::open_light_ui()
{
    light_ui_->show();
    this->hide();
}

// Set default preset from preset.json (called from .ui)
void MainWindow::set_preset()
{
    std::filesystem::path preset_directory_path(RELATIVE_PATH(__PRESET_FOLDER_PATH__ / "doppler_8b_384_384_27.json"));
    reload_ini(preset_directory_path.string());
    LOG_INFO("Preset loaded");
}

// Set preset from a file (called on notify)
void MainWindow::set_preset(std::filesystem::path file)
{
    reload_ini(file.string());
    LOG_INFO("Preset loaded with file " + file.string());
}

void MainWindow::init_tooltips()
{
    ui_->ViewModeComboBox->setItemData(static_cast<int>(ImgType::Modulus),
                                       "Compute the modulus of the complex image",
                                       Qt::ToolTipRole);
    ui_->ViewModeComboBox->setItemData(static_cast<int>(ImgType::SquaredModulus),
                                       "Compute the squared modulus of the complex image",
                                       Qt::ToolTipRole);
    ui_->ViewModeComboBox->setItemData(static_cast<int>(ImgType::Argument),
                                       "Compute the argument angle of the complex image",
                                       Qt::ToolTipRole);
    ui_->ViewModeComboBox->setItemData(static_cast<int>(ImgType::PhaseIncrease),
                                       "Adjust phase angles of the complex image",
                                       Qt::ToolTipRole);
    ui_->ViewModeComboBox->setItemData(static_cast<int>(ImgType::Composite),
                                       "Do computations with color encoding",
                                       Qt::ToolTipRole);

    ui_->SpaceTransformationComboBox->setItemData(static_cast<int>(SpaceTransformation::NONE),
                                                  "No space transformation",
                                                  Qt::ToolTipRole);
    ui_->SpaceTransformationComboBox->setItemData(static_cast<int>(SpaceTransformation::FRESNELTR),
                                                  "Fresnel Transform",
                                                  Qt::ToolTipRole);
    ui_->SpaceTransformationComboBox->setItemData(static_cast<int>(SpaceTransformation::ANGULARSP),
                                                  "Angular Spectrum Propagation",
                                                  Qt::ToolTipRole);

    ui_->TimeTransformationComboBox->setItemData(static_cast<int>(TimeTransformation::NONE),
                                                 "No time transformation",
                                                 Qt::ToolTipRole);
    ui_->TimeTransformationComboBox->setItemData(static_cast<int>(TimeTransformation::PCA),
                                                 "Principal Component Analysis",
                                                 Qt::ToolTipRole);
    ui_->TimeTransformationComboBox->setItemData(static_cast<int>(TimeTransformation::STFT),
                                                 "Short-Time Fourier Transformation",
                                                 Qt::ToolTipRole);
    ui_->TimeTransformationComboBox->setItemData(static_cast<int>(TimeTransformation::SSA_STFT),
                                                 "Self-adaptive Spectrum Analysis Short-Time Fourier Transformation",
                                                 Qt::ToolTipRole);
}

#pragma endregion

#pragma region Themes

void MainWindow::set_night()
{
    theme_ = Theme::Dark;
    QFile file(":/assets/style/style.css");
    file.open(QFile::ReadOnly);
    QString styleSheet = QLatin1String(file.readAll());

    qApp->setStyleSheet(styleSheet);
}

void MainWindow::set_classic()
{
    qApp->setPalette(this->style()->standardPalette());
    qApp->setStyleSheet("");
    theme_ = Theme::Classic;
}

void MainWindow::set_theme(const Theme theme)
{
    qApp->setStyle(QStyleFactory::create("Fusion"));

    if (theme == Theme::Classic)
        set_classic();
    else
        set_night();
}
#pragma endregion
} // namespace holovibes::gui
