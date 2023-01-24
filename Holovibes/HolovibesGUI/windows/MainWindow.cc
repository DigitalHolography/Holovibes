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

#include "API.hh"

#include "view_struct.hh"

#include "asw_mainwindow_panel.hh"

#include "user_interface.hh"

#define MIN_IMG_NB_TIME_TRANSFORMATION_CUTS 8

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
{
    UserInterface::instance().main_window = this;

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

    setWindowIcon(QIcon(":/holovibes_logo.png"));

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

    load_gui();

    try
    {
        api::load_compute_settings(holovibes::settings::compute_settings_filepath);
        // Set values not set by notify
        ui_->BatchSizeSpinBox->setValue(api::get_batch_size());
    }
    catch (const std::exception&)
    {
        LOG_INFO("{}: Compute settings file not found. Initialization with default values.",
                 ::holovibes::settings::compute_settings_filepath);
        api::save_compute_settings(holovibes::settings::compute_settings_filepath);
    }

    // Display default values
    api::set_compute_mode(ComputeModeEnum::Raw);

    setFocusPolicy(Qt::StrongFocus);

    // spinBox allow ',' and '.' as decimal point
    spinBoxDecimalPointReplacement(ui_->WaveLengthDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui_->ZDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui_->ContrastMaxDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui_->ContrastMinDoubleSpinBox);

    // TODO: move in AppData
    // Fill the quick kernel combo box with files from convolution_kernels
    // directory
    std::filesystem::path convo_matrix_path(get_exe_dir());
    convo_matrix_path = convo_matrix_path / "convolution_kernels";
    if (std::filesystem::exists(convo_matrix_path))
    {
        QVector<QString> files;
        files.push_back(QString(UID_CONVOLUTION_TYPE_DEFAULT));
        for (const auto& file : std::filesystem::directory_iterator(convo_matrix_path))
            files.push_back(QString(file.path().filename().string().c_str()));
        std::sort(files.begin(), files.end(), [&](const auto& a, const auto& b) { return a < b; });
        ui_->KernelQuickSelectComboBox->addItems(QStringList::fromVector(files));
    }

    // Initialize all panels
    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->init();

    qApp->setStyle(QStyleFactory::create("Fusion"));
}

MainWindow::~MainWindow() { delete ui_; }

#pragma endregion
/* ------------ */
#pragma region Notify
void MainWindow::synchronize_thread(std::function<void()> f, bool sync)
{
    if (QThread::currentThread() == this->thread())
        return f();

    if (sync == false)
    {
        emit synchronize_thread_signal(f);
        return;
    }

    std::atomic<bool> has_finish = false;
    emit synchronize_thread_signal(
        [&]()
        {
            f();
            has_finish = true;
        });

    while (has_finish == false)
        ;
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

    ui_->actionSettings->setEnabled(true);

    // Tabs
    if (api::get_import_type() == ImportTypeEnum::None)
    {
        ui_->CompositePanel->hide();
        ui_->ImageRenderingPanel->setEnabled(false);
        ui_->ViewPanel->setEnabled(false);
        ui_->ExportPanel->setEnabled(false);
        layout_toggled();
        return;
    }

    ui_->ImageRenderingPanel->setEnabled(true);
    ui_->ViewPanel->setEnabled(api::get_compute_mode() == ComputeModeEnum::Hologram);
    ui_->ExportPanel->setEnabled(true);
    ui_->CompositePanel->setHidden(api::get_compute_mode() == ComputeModeEnum::Raw ||
                                   (api::get_image_type() != ImageTypeEnum::Composite));
    resize(baseSize());
    adjustSize();
}

static void handle_accumulation_exception() { api::change_view_xy()->output_image_accumulation = 1; }

void MainWindow::notify_error(const std::exception& e)
{
    const CustomException* err_ptr = dynamic_cast<const CustomException*>(&e);
    if (err_ptr)
    {
        const UpdateException* err_update_ptr = dynamic_cast<const UpdateException*>(err_ptr);
        if (err_update_ptr)
        {
            auto lambda = [&]
            {
                // notify will be in close_critical_compute
                api::change_view_accu_p()->start = 0;
                api::set_time_transformation_size(1);
                api::set_import_type(ImportTypeEnum::None);
                LOG_ERROR("GPU computing error occured. : {}", e.what());
            };
            synchronize_thread(lambda);
        }

        auto lambda = [&, accu = (dynamic_cast<const AccumulationException*>(err_ptr) != nullptr)]
        {
            if (accu)
                handle_accumulation_exception();
            api::set_import_type(ImportTypeEnum::None);
            LOG_ERROR("GPU computing error occured. : {}", e.what());
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
    const std::string msg = api::get_credits();

    // Creation on the fly of the message box to display
    QMessageBox msg_box;
    msg_box.setText(QString::fromUtf8(msg.c_str()));
    msg_box.setIcon(QMessageBox::Information);
    msg_box.exec();
}

void MainWindow::documentation() { QDesktopServices::openUrl(QUrl(api::get_documentation_url().c_str())); }

#pragma endregion
/* ------------ */
#pragma region Json

void MainWindow::write_compute_settings()
{
    api::save_compute_settings(holovibes::settings::compute_settings_filepath);
}

void MainWindow::browse_export_ini()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("All files (*.json)"));
    api::save_compute_settings(filename.toStdString());
}

void MainWindow::reload_ini(const std::string& filename)
{
    ImportTypeEnum it = api::get_import_type();
    ui_->ImportPanel->import_stop();

    api::load_compute_settings(filename);

    // Set values not set by notify
    ui_->BatchSizeSpinBox->setValue(api::get_batch_size());

    if (it == ImportTypeEnum::File)
        ui_->ImportPanel->import_start();
    else if (it == ImportTypeEnum::Camera)
        change_camera(api::get_current_camera_kind());
    else if (it == ImportTypeEnum::None)
        notify();
}

void MainWindow::browse_import_ini()
{
    QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("import .json file"),
                                                    UserInterface::instance().file_input_directory_.c_str(),
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
        return;
    }

    set_theme(json_get_or_default(j_us, Theme::Dark, "display", "theme"));

    UserInterface::instance().window_max_size =
        json_get_or_default(j_us, UserInterface::instance().window_max_size, "windows", "main window max size");
    UserInterface::instance().auxiliary_window_max_size =
        json_get_or_default(j_us,
                            UserInterface::instance().auxiliary_window_max_size,
                            "windows",
                            "auxiliary window max size");

    api::set_display_rate(json_get_or_default(j_us, api::get_display_rate(), "display", "refresh rate"));
    api::set_raw_bitshift(json_get_or_default(j_us, api::get_raw_bitshift(), "file info", "raw bit shift"));

    ui_->ExportPanel->set_record_frame_step(
        json_get_or_default(j_us, ui_->ExportPanel->get_record_frame_step(), "gui settings", "record frame step"));

    UserInterface::instance().auto_scale_point_threshold_ =
        json_get_or_default(j_us,
                            UserInterface::instance().auto_scale_point_threshold_,
                            "chart",
                            "auto scale point threshold");
    UserInterface::instance().default_output_filename_ =
        json_get_or_default(j_us,
                            UserInterface::instance().default_output_filename_,
                            "files",
                            "default output filename");
    UserInterface::instance().record_output_directory_ =
        json_get_or_default(j_us,
                            UserInterface::instance().record_output_directory_,
                            "files",
                            "record output directory");
    UserInterface::instance().file_input_directory_ =
        json_get_or_default(j_us, UserInterface::instance().file_input_directory_, "files", "file input directory");
    UserInterface::instance().batch_input_directory_ =
        json_get_or_default(j_us, UserInterface::instance().batch_input_directory_, "files", "batch input directory");

    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->load_gui(j_us);

    notify();
}

void MainWindow::save_gui()
{
    if (holovibes::settings::user_settings_filepath.empty())
        return;

    json j_us;

    j_us["display"]["theme"] = theme_;

    j_us["windows"]["main window max size"] = UserInterface::window_max_size;
    j_us["windows"]["auxiliary window max size"] = UserInterface::auxiliary_window_max_size;

    j_us["display"]["refresh rate"] = api::get_display_rate();
    j_us["file info"]["raw bit shift"] = api::get_raw_bitshift();
    j_us["gui settings"]["record frame step"] = ui_->ExportPanel->get_record_frame_step();
    j_us["chart"]["auto scale point threshold"] = UserInterface::instance().auto_scale_point_threshold_;
    j_us["files"]["default output filename"] = UserInterface::instance().default_output_filename_;
    j_us["files"]["record output directory"] = UserInterface::instance().record_output_directory_;
    j_us["files"]["file input directory"] = UserInterface::instance().file_input_directory_;
    j_us["files"]["batch input directory"] = UserInterface::instance().batch_input_directory_;

    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->save_gui(j_us);

    auto path = holovibes::settings::user_settings_filepath;
    std::ofstream file(path);
    file << j_us.dump(1);

    LOG_INFO("user settings overwritten at {}", path);
}

#pragma endregion
/* ------------ */
#pragma region Close Compute

void MainWindow::closeEvent(QCloseEvent*)
{
    api::detail::set_value<ImportType>(ImportTypeEnum::None);

    save_gui();
    if (save_cs)
        api::save_compute_settings();
}

#pragma endregion
/* ------------ */
#pragma region Cameras

void MainWindow::change_camera(CameraKind c) { api::set_current_camera_kind(c); }

void MainWindow::camera_none() { change_camera(CameraKind::None); }

void MainWindow::camera_ids() { change_camera(CameraKind::IDS); }

void MainWindow::camera_phantom() { change_camera(CameraKind::Phantom); }

void MainWindow::camera_bitflow_cyton() { change_camera(CameraKind::BitflowCyton); }

void MainWindow::camera_hamamatsu() { change_camera(CameraKind::Hamamatsu); }

void MainWindow::camera_adimec() { change_camera(CameraKind::Adimec); }

void MainWindow::camera_xiq() { change_camera(CameraKind::xiQ); }

void MainWindow::camera_xib() { change_camera(CameraKind::xiB); }

void MainWindow::camera_opencv() { change_camera(CameraKind::OpenCV); }

void MainWindow::configure_camera()
{
    if (Holovibes::instance().get_active_camera() != nullptr)
        QDesktopServices::openUrl(
            QUrl::fromLocalFile(QString::fromStdString(Holovibes::instance().get_active_camera()->get_ini_name())));
}
#pragma endregion
/* ------------ */
#pragma region Image Mode

void MainWindow::refresh_view_mode() {}

void MainWindow::set_composite_values()
{
    const unsigned min_val_composite = api::get_time_transformation_size() == 1 ? 0 : 1;
    const unsigned max_val_composite = api::get_time_transformation_size() - 1;

    ui_->PRedSpinBox_Composite->setValue(min_val_composite);
    ui_->SpinBox_hue_freq_min->setValue(min_val_composite);
    ui_->SpinBox_saturation_freq_min->setValue(min_val_composite);
    ui_->SpinBox_value_freq_min->setValue(min_val_composite);

    ui_->PBlueSpinBox_Composite->setValue(max_val_composite);
    ui_->SpinBox_hue_freq_max->setValue(max_val_composite);
    ui_->SpinBox_saturation_freq_max->setValue(max_val_composite);
    ui_->SpinBox_value_freq_max->setValue(max_val_composite);
}

void MainWindow::set_view_image_type(const QString& value)
{
    api::set_image_type(static_cast<ImageTypeEnum>(ui_->ViewModeComboBox->currentIndex()));
}

#pragma endregion

/* ------------ */

Ui::MainWindow* MainWindow::get_ui() { return ui_; }

#pragma endregion
/* ------------ */
#pragma region Advanced

void MainWindow::close_advanced_settings()
{
    if (UserInterface::instance().has_been_updated)
    {
        ImportTypeEnum it = api::get_import_type();
        ui_->ImportPanel->import_stop();

        if (it == ImportTypeEnum::File)
            ui_->ImportPanel->import_start();
        else if (it == ImportTypeEnum::Camera)
            change_camera(api::get_current_camera_kind());
    }

    UserInterface::instance().is_advanced_settings_displayed = false;
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

void MainWindow::open_advanced_settings()
{
    ASWMainWindowPanel* panel = new ASWMainWindowPanel(this);

    UserInterface::instance().is_advanced_settings_displayed = true;
    UserInterface::instance().advanced_settings_window =
        std::make_unique<::holovibes::gui::AdvancedSettingsWindow>(this, panel);

    connect(UserInterface::instance().advanced_settings_window.get(),
            SIGNAL(closed()),
            this,
            SLOT(close_advanced_settings()),
            Qt::UniqueConnection);
}

#pragma endregion

/* ------------ */
#pragma region UI

void MainWindow::shift_screen()
{
    // shift main window when camera view appears
    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();
    move(QPoint(210 + (screen_width - 800) / 2, 200 + (screen_height - 500) / 2));
}

#pragma endregion

/* ------------ */
#pragma region Themes

void MainWindow::set_night()
{
    // Dark mode style
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
    darkPalette.setColor(QPalette::PlaceholderText, Qt::darkGray);
    darkPalette.setColor(QPalette::Link, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::Highlight, QColor(42, 130, 218));
    darkPalette.setColor(QPalette::HighlightedText, Qt::black);
    darkPalette.setColor(QPalette::Light, Qt::black);

    qApp->setPalette(darkPalette);
    theme_ = Theme::Dark;
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
