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
#include "ini_config.hh"
#include "update_exception.hh"
#include "accumulation_exception.hh"
#include "gui_group_box.hh"

#include "API.hh"

#include "view_struct.hh"

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
MainWindow::MainWindow(QWidget* parent)
    : QMainWindow(parent)
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

    ::holovibes::worker::InformationWorker::display_info_text_function_ = [=](const std::string& text) {
        synchronize_thread([=]() { ui_->InfoPanel->set_text(text.c_str()); });
    };

    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();

    // need the correct dimensions of main windows
    move(QPoint((screen_width - 800) / 2, (screen_height - 500) / 2));

    // Set default files
    std::filesystem::path holovibes_documents_path = get_user_documents_path() / __APPNAME__;
    std::filesystem::create_directory(holovibes_documents_path);

    try
    {
        load_gui();
    }
    catch (const std::exception&)
    {
        LOG_INFO << ::holovibes::ini::global_config_filepath << ": global configuration file not found. "
                 << "Initialization with default values.";
        save_gui();
    }

    try
    {
        api::load_compute_settings(holovibes::ini::default_compute_config_filepath);
    }
    catch (const std::exception&)
    {
        LOG_INFO << ::holovibes::ini::default_compute_config_filepath << ": Configuration file not found. "
                 << "Initialization with default values.";
        api::save_compute_settings(holovibes::ini::default_compute_config_filepath);
    }

    // Display default values
    api::set_compute_mode(Computation::Raw);
    UserInterfaceDescriptor::instance().last_img_type_ = api::get_img_type() == ImgType::Composite
                                                             ? "Composite image"
                                                             : UserInterfaceDescriptor::instance().last_img_type_;
    notify();

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

    api::start_information_display();
}

MainWindow::~MainWindow()
{
    api::close_windows();
    api::close_critical_compute();
    api::stop_all_worker_controller();

    camera_none();

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
    if (api::get_is_computation_stopped())
    {
        ui_->CompositePanel->hide();
        ui_->ImageRenderingPanel->setEnabled(false);
        ui_->ViewPanel->setEnabled(false);
        ui_->ExportPanel->setEnabled(false);
        layout_toggled();
        return;
    }

    if (UserInterfaceDescriptor::instance().is_enabled_camera_)
    {
        ui_->ImageRenderingPanel->setEnabled(true);
        ui_->ViewPanel->setEnabled(api::get_compute_mode() == Computation::Hologram);
        ui_->ExportPanel->setEnabled(true);
    }

    ui_->CompositePanel->setHidden(api::is_raw_mode() || (api::get_cd().img_type != ImgType::Composite));
}

void MainWindow::notify_error(const std::exception& e)
{
    const CustomException* err_ptr = dynamic_cast<const CustomException*>(&e);
    if (err_ptr)
    {
        const UpdateException* err_update_ptr = dynamic_cast<const UpdateException*>(err_ptr);
        if (err_update_ptr)
        {
            auto lambda = [&, this] {
                // notify will be in close_critical_compute
                api::get_cd().handle_update_exception();
                api::close_windows();
                api::close_critical_compute();
                LOG_ERROR << "GPU computing error occured.";
                LOG_ERROR << e.what();
                notify();
            };
            synchronize_thread(lambda);
        }

        auto lambda = [&, this, accu = (dynamic_cast<const AccumulationException*>(err_ptr) != nullptr)] {
            if (accu)
            {
                api::get_cd().handle_accumulation_exception();
            }
            api::close_critical_compute();

            LOG_ERROR << "GPU computing error occured.";
            LOG_ERROR << e.what();
            notify();
        };
        synchronize_thread(lambda);
    }
    else
    {
        LOG_ERROR << "Unknown error occured.";
        LOG_ERROR << e.what();
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
    const std::string msg = api::get_credits();

    // Creation on the fly of the message box to display
    QMessageBox msg_box;
    msg_box.setText(QString::fromUtf8(msg.c_str()));
    msg_box.setIcon(QMessageBox::Information);
    msg_box.exec();
}

void MainWindow::documentation() { QDesktopServices::openUrl(api::get_documentation_url()); }

#pragma endregion
/* ------------ */
#pragma region Ini

void MainWindow::write_ini() { api::save_compute_settings(); }

void MainWindow::browse_export_ini()
{
    QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("All files (*.ini)"));
    api::save_compute_settings(filename.toStdString());
}

void MainWindow::reload_ini(const std::string& filename)
{
    // May be removed because it is the first call of import_start call just after.
    ui_->ImportPanel->import_stop();

    api::load_compute_settings(filename);

    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::File)
        ui_->ImportPanel->import_start();
    else if (UserInterfaceDescriptor::instance().import_type_ == ImportType::Camera)
        change_camera(UserInterfaceDescriptor::instance().kCamera);

    notify();
}

void MainWindow::browse_import_ini()
{
    QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("import .ini file"),
                                                    UserInterfaceDescriptor::instance().file_input_directory_.c_str(),
                                                    tr("All files (*.ini);; Ini files (*.ini)"));

    if (!filename.isEmpty())
        reload_ini(filename.toStdString());
}

void MainWindow::reload_ini() { reload_ini(::holovibes::ini::default_compute_config_filepath); }

void set_module_visibility(QAction*& action, GroupBox*& groupbox, bool to_hide)
{
    action->setChecked(!to_hide);
    groupbox->setHidden(to_hide);
}

void MainWindow::load_gui()
{
    boost::property_tree::ptree ptree;
    boost::property_tree::ini_parser::read_ini(ini::global_config_filepath, ptree);

    if (!ptree.empty())
    {
        set_theme(ptree.get<int>("display.theme_type", theme_index_));

        window_max_size = ptree.get<uint>("window_size.main_window_max_size", window_max_size);
        auxiliary_window_max_size = ptree.get<uint>("window_size.auxiliary_window_max_size", 512);

        api::load_user_preferences(ptree);

        for (auto it = panels_.begin(); it != panels_.end(); it++)
            (*it)->load_gui(ptree);

        notify();
    }
}

void MainWindow::save_gui()
{
    boost::property_tree::ptree ptree;

    ptree.put<ushort>("display.theme_type", theme_index_);

    ptree.put<uint>("window_size.main_window_max_size", window_max_size);
    ptree.put<uint>("window_size.auxiliary_window_max_size", auxiliary_window_max_size);

    api::save_user_preferences(ptree);

    for (auto it = panels_.begin(); it != panels_.end(); it++)
        (*it)->save_gui(ptree);

    auto path = holovibes::ini::global_config_filepath;
    boost::property_tree::write_ini(path, ptree);

    LOG_INFO << " GUI settings overwritten at " << path;
}

#pragma endregion
/* ------------ */
#pragma region Close Compute

void MainWindow::closeEvent(QCloseEvent*)
{
    camera_none();

    save_gui();
    api::save_compute_settings();
}

#pragma endregion
/* ------------ */
#pragma region Cameras

void MainWindow::change_camera(CameraKind c)
{
    // Weird call to setup none camera before changing
    camera_none();

    if (c == CameraKind::NONE)
        return;

    const bool res = api::change_camera(c);

    if (res)
    {
        // Shows Holo/Raw window
        ui_->ImageRenderingPanel->set_image_mode(static_cast<int>(api::get_compute_mode()));
        shift_screen();

        // Make camera's settings menu accessible
        QAction* settings = ui_->actionSettings;
        settings->setEnabled(true);

        notify();
    }
}

void MainWindow::camera_none()
{
    api::camera_none();

    // Make camera's settings menu unaccessible
    ui_->actionSettings->setEnabled(false);

    notify();
}

void MainWindow::camera_ids() { change_camera(CameraKind::IDS); }

void MainWindow::camera_phantom() { change_camera(CameraKind::Phantom); }

void MainWindow::camera_bitflow_cyton() { change_camera(CameraKind::BitflowCyton); }

void MainWindow::camera_hamamatsu() { change_camera(CameraKind::Hamamatsu); }

void MainWindow::camera_adimec() { change_camera(CameraKind::Adimec); }

void MainWindow::camera_xiq() { change_camera(CameraKind::xiQ); }

void MainWindow::camera_xib() { change_camera(CameraKind::xiB); }

void MainWindow::configure_camera() { api::configure_camera(); }
#pragma endregion
/* ------------ */
#pragma region Image Mode

void MainWindow::refresh_view_mode()
{
    // FIXME: Create enum instead of using index.
    api::refresh_view_mode(*this, window_max_size, ui_->ViewModeComboBox->currentIndex());

    notify();
    layout_toggled();
}

// Is there a change in window pixel depth (needs to be re-opened)
bool MainWindow::need_refresh(const std::string& last_type, const std::string& new_type)
{
    std::vector<std::string> types_needing_refresh({"Composite image"});
    for (auto& type : types_needing_refresh)
        if ((last_type == type) != (new_type == type))
            return true;
    return false;
}

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
    if (api::is_raw_mode())
        return;

    const std::string& str = value.toStdString();

    if (need_refresh(UserInterfaceDescriptor::instance().last_img_type_, str))
    {
        refresh_view_mode();
        if (api::get_img_type() == ImgType::Composite)
        {
            set_composite_values();
        }
    }

    auto callback = ([=]() {
        api::set_img_type(static_cast<ImgType>(ui_->ViewModeComboBox->currentIndex()));
        notify();
        layout_toggled();
    });

    // Force XYview autocontrast
    api::set_view_mode(str, callback);

    // Force cuts views autocontrast if needed
    if (api::get_3d_cuts_view_enabled())
        api::set_auto_contrast_cuts();
}

#pragma endregion

/* ------------ */

void MainWindow::change_window(int index)
{
    api::change_window(index);

    notify();
}

void MainWindow::start_import(QString filename)
{
    ui_->ImportPanel->import_file(filename);
    ui_->ImportPanel->import_start();
}

Ui::MainWindow* MainWindow::get_ui() { return ui_; }

#pragma endregion
/* ------------ */
#pragma region Advanced

void MainWindow::close_advanced_settings()
{
    if (UserInterfaceDescriptor::instance().need_close)
        close();
    else
        UserInterfaceDescriptor::instance().is_advanced_settings_displayed = false;
}

void MainWindow::open_advanced_settings()
{
    if (UserInterfaceDescriptor::instance().is_advanced_settings_displayed)
        return;

    api::open_advanced_settings();

    connect(UserInterfaceDescriptor::instance().advanced_settings_window_.get(),
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

void MainWindow::set_theme(const int index)
{
    if (index == theme_index_)
        return;

    switch (index)
    {
    case 0:
        set_classic();
        break;
    case 1:
        set_night();
        break;
    default:
        return;
    }

    theme_index_ = index;
}
#pragma endregion
} // namespace gui
} // namespace holovibes
