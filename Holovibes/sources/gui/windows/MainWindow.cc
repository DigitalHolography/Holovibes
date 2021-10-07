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
{
    ui.setupUi(this);

    qRegisterMetaType<std::function<void()>>();
    connect(this,
            SIGNAL(synchronize_thread_signal(std::function<void()>)),
            this,
            SLOT(synchronize_thread(std::function<void()>)));

    setWindowIcon(QIcon("Holovibes.ico"));

    auto display_info_text_fun = [=](const std::string& text) {
        synchronize_thread([=]() { ui.InfoPanel->set_text(text.c_str()); });
    };
    Holovibes::instance().get_info_container().set_display_info_text_function(display_info_text_fun);

    auto update_progress = [=](InformationContainer::ProgressType type, const size_t value, const size_t max_size) {
        synchronize_thread([=]() {
            switch (type)
            {
            case InformationContainer::ProgressType::FILE_READ:
                ui.InfoPanel->init_file_reader_progress(static_cast<int>(value), static_cast<int>(max_size));
                break;
            case InformationContainer::ProgressType::CHART_RECORD:
            case InformationContainer::ProgressType::FRAME_RECORD:
                ui.InfoPanel->init_record_progress(static_cast<int>(value), static_cast<int>(max_size));
                break;
            default:
                return;
            };
        });
    };
    Holovibes::instance().get_info_container().set_update_progress_function(update_progress);
    ui.InfoPanel->set_visible_file_reader_progress(false);
    ui.InfoPanel->set_visible_record_progress(false);

    ui.ExportPanel->set_record_mode(QString::fromUtf8("Raw Image"));

    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();

    // need the correct dimensions of main windows
    move(QPoint((screen_width - 800) / 2, (screen_height - 500) / 2));

    // Hide non default tab
    ui.CompositePanel->hide();

    // Set default files
    std::filesystem::path holovibes_documents_path = get_user_documents_path() / "Holovibes";
    std::filesystem::create_directory(holovibes_documents_path);
    default_output_filename_ = "capture";
    record_output_directory_ = holovibes_documents_path.string();
    file_input_directory_ = "C:\\";
    batch_input_directory_ = "C:\\";

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

    set_z_step(z_step_);
    ui.ExportPanel->set_record_frame_step(record_frame_step_);
    set_night();

    // Keyboard shortcuts
    z_up_shortcut_ = new QShortcut(QKeySequence("Up"), this);
    z_up_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(z_up_shortcut_, SIGNAL(activated()), this, SLOT(increment_z()));

    z_down_shortcut_ = new QShortcut(QKeySequence("Down"), this);
    z_down_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(z_down_shortcut_, SIGNAL(activated()), this, SLOT(decrement_z()));

    p_left_shortcut_ = new QShortcut(QKeySequence("Left"), this);
    p_left_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_left_shortcut_, SIGNAL(activated()), ui.ViewPanel, SLOT(decrement_p()));

    p_right_shortcut_ = new QShortcut(QKeySequence("Right"), this);
    p_right_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_right_shortcut_, SIGNAL(activated()), ui.ViewPanel, SLOT(increment_p()));

    QComboBox* window_cbox = ui.WindowSelectionComboBox;
    connect(window_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(change_window()));

    // Display default values
    cd_.set_compute_mode(Computation::Raw);
    notify();
    setFocusPolicy(Qt::StrongFocus);

    // spinBox allow ',' and '.' as decimal point
    spinBoxDecimalPointReplacement(ui.WaveLengthDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui.ZDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui.ContrastMaxDoubleSpinBox);
    spinBoxDecimalPointReplacement(ui.ContrastMinDoubleSpinBox);

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
        ui.KernelQuickSelectComboBox->addItems(QStringList::fromVector(files));
    }

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
    ui.InputBrowseToolButton->setEnabled(cd_.is_computation_stopped);

    // Tabs
    if (cd_.is_computation_stopped)
    {
        ui.CompositePanel->hide();
        ui.ImageRenderingGroupBox->setEnabled(false);
        ui.ViewPanel->setEnabled(false);
        ui.ExportPanel->setEnabled(false);
        layout_toggled();
        return;
    }

    if (is_enabled_camera_)
    {
        ui.ImageRenderingGroupBox->setEnabled(true);
        ui.ViewPanel->setEnabled(cd_.compute_mode == Computation::Hologram);
        ui.ExportPanel->setEnabled(true);
    }

    const bool is_raw = is_raw_mode();

    if (is_raw)
    {
        ui.RecordImageModeComboBox->removeItem(ui.RecordImageModeComboBox->findText("Processed Image"));
        ui.RecordImageModeComboBox->removeItem(ui.RecordImageModeComboBox->findText("Chart"));
    }
    else // Hologram mode
    {
        if (ui.RecordImageModeComboBox->findText("Processed Image") == -1)
            ui.RecordImageModeComboBox->insertItem(1, "Processed Image");
        if (ui.RecordImageModeComboBox->findText("Chart") == -1)
            ui.RecordImageModeComboBox->insertItem(2, "Chart");
    }

    // Raw view
    ui.RawDisplayingCheckBox->setEnabled(!is_raw);
    ui.RawDisplayingCheckBox->setChecked(!is_raw && cd_.raw_view_enabled);

    QPushButton* signalBtn = ui.ChartSignalPushButton;
    signalBtn->setStyleSheet(
        (mainDisplay && signalBtn->isEnabled() && mainDisplay->getKindOfOverlay() == KindOfOverlay::Signal)
            ? "QPushButton {color: #8E66D9;}"
            : "");

    QPushButton* noiseBtn = ui.ChartNoisePushButton;
    noiseBtn->setStyleSheet(
        (mainDisplay && noiseBtn->isEnabled() && mainDisplay->getKindOfOverlay() == KindOfOverlay::Noise)
            ? "QPushButton {color: #00A4AB;}"
            : "");

    ui.PhaseUnwrap2DCheckBox->setEnabled(cd_.img_type == ImgType::PhaseIncrease || cd_.img_type == ImgType::Argument);

    // Time transformation cuts
    ui.TimeTransformationCutsCheckBox->setChecked(!is_raw && cd_.time_transformation_cuts_enabled);

    // Contrast
    ui.ContrastCheckBox->setChecked(!is_raw && cd_.contrast_enabled);
    ui.ContrastCheckBox->setEnabled(true);
    ui.AutoRefreshContrastCheckBox->setChecked(cd_.contrast_auto_refresh);

    // Contrast SpinBox:
    ui.ContrastMinDoubleSpinBox->setEnabled(!cd_.contrast_auto_refresh);
    ui.ContrastMinDoubleSpinBox->setValue(cd_.get_contrast_min());
    ui.ContrastMaxDoubleSpinBox->setEnabled(!cd_.contrast_auto_refresh);
    ui.ContrastMaxDoubleSpinBox->setValue(cd_.get_contrast_max());

    // FFT shift
    ui.FFTShiftCheckBox->setChecked(cd_.fft_shift_enabled);
    ui.FFTShiftCheckBox->setEnabled(true);

    // Window selection
    QComboBox* window_selection = ui.WindowSelectionComboBox;
    window_selection->setEnabled(cd_.time_transformation_cuts_enabled);
    window_selection->setCurrentIndex(window_selection->isEnabled() ? static_cast<int>(cd_.current_window.load()) : 0);

    ui.LogScaleCheckBox->setEnabled(true);
    ui.LogScaleCheckBox->setChecked(!is_raw && cd_.get_img_log_scale_slice_enabled(cd_.current_window.load()));
    ui.ImgAccuCheckBox->setEnabled(true);
    ui.ImgAccuCheckBox->setChecked(!is_raw && cd_.get_img_acc_slice_enabled(cd_.current_window.load()));
    ui.ImgAccuSpinBox->setValue(cd_.get_img_acc_slice_level(cd_.current_window.load()));
    if (cd_.current_window == WindowKind::XYview)
    {
        ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(displayAngle))).c_str());
        ui.FlipPushButton->setText(("Flip " + std::to_string(displayFlip)).c_str());
    }
    else if (cd_.current_window == WindowKind::XZview)
    {
        ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(xzAngle))).c_str());
        ui.FlipPushButton->setText(("Flip " + std::to_string(xzFlip)).c_str());
    }
    else if (cd_.current_window == WindowKind::YZview)
    {
        ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(yzAngle))).c_str());
        ui.FlipPushButton->setText(("Flip " + std::to_string(yzFlip)).c_str());
    }

    // p accu
    ui.PAccuCheckBox->setEnabled(cd_.img_type != ImgType::PhaseIncrease);
    ui.PAccuCheckBox->setChecked(cd_.p_accu_enabled);
    ui.PAccSpinBox->setMaximum(cd_.time_transformation_size - 1);

    cd_.check_p_limits();
    ui.PAccSpinBox->setValue(cd_.p_acc_level);
    ui.PSpinBox->setValue(cd_.pindex);
    ui.PAccSpinBox->setEnabled(cd_.img_type != ImgType::PhaseIncrease);
    if (cd_.p_accu_enabled)
    {
        ui.PSpinBox->setMaximum(cd_.time_transformation_size - cd_.p_acc_level - 1);
        ui.PAccSpinBox->setMaximum(cd_.time_transformation_size - cd_.pindex - 1);
    }
    else
    {
        ui.PSpinBox->setMaximum(cd_.time_transformation_size - 1);
    }
    ui.PSpinBox->setEnabled(!is_raw);

    // q accu
    bool is_ssa_stft = cd_.time_transformation == TimeTransformation::SSA_STFT;
    ui.Q_AccuCheckBox->setEnabled(is_ssa_stft && !is_raw);
    ui.Q_AccSpinBox->setEnabled(is_ssa_stft && !is_raw);
    ui.Q_SpinBox->setEnabled(is_ssa_stft && !is_raw);

    ui.Q_AccuCheckBox->setChecked(cd_.q_acc_enabled);
    ui.Q_AccSpinBox->setMaximum(cd_.time_transformation_size - 1);

    cd_.check_q_limits();
    ui.Q_AccSpinBox->setValue(cd_.q_acc_level);
    ui.Q_SpinBox->setValue(cd_.q_index);
    if (cd_.q_acc_enabled)
    {
        ui.Q_SpinBox->setMaximum(cd_.time_transformation_size - cd_.q_acc_level - 1);
        ui.Q_AccSpinBox->setMaximum(cd_.time_transformation_size - cd_.q_index - 1);
    }
    else
    {
        ui.Q_SpinBox->setMaximum(cd_.time_transformation_size - 1);
    }

    // XY accu
    ui.XAccuCheckBox->setChecked(cd_.x_accu_enabled);
    ui.XAccSpinBox->setValue(cd_.x_acc_level);
    ui.YAccuCheckBox->setChecked(cd_.y_accu_enabled);
    ui.YAccSpinBox->setValue(cd_.y_acc_level);

    int max_width = 0;
    int max_height = 0;
    if (holovibes_.get_gpu_input_queue() != nullptr)
    {
        max_width = holovibes_.get_gpu_input_queue()->get_fd().width - 1;
        max_height = holovibes_.get_gpu_input_queue()->get_fd().height - 1;
    }
    else
    {
        cd_.x_cuts = 0;
        cd_.y_cuts = 0;
    }
    ui.XSpinBox->setMaximum(max_width);
    ui.YSpinBox->setMaximum(max_height);
    QSpinBoxQuietSetValue(ui.XSpinBox, cd_.x_cuts);
    QSpinBoxQuietSetValue(ui.YSpinBox, cd_.y_cuts);

    // Time transformation
    ui.TimeTransformationStrideSpinBox->setEnabled(!is_raw);

    const uint input_queue_capacity = global::global_config.input_queue_max_size;

    ui.TimeTransformationStrideSpinBox->setValue(cd_.time_transformation_stride);
    ui.TimeTransformationStrideSpinBox->setSingleStep(cd_.batch_size);
    ui.TimeTransformationStrideSpinBox->setMinimum(cd_.batch_size);

    // Batch
    ui.BatchSizeSpinBox->setEnabled(!is_raw && !is_recording_);

    cd_.check_batch_size_limit(input_queue_capacity);
    ui.BatchSizeSpinBox->setValue(cd_.batch_size);
    ui.BatchSizeSpinBox->setMaximum(input_queue_capacity);

    // Image rendering
    ui.SpaceTransformationComboBox->setEnabled(!is_raw && !cd_.time_transformation_cuts_enabled);
    ui.SpaceTransformationComboBox->setCurrentIndex(static_cast<int>(cd_.space_transformation.load()));
    ui.TimeTransformationComboBox->setEnabled(!is_raw);
    ui.TimeTransformationComboBox->setCurrentIndex(static_cast<int>(cd_.time_transformation.load()));

    // Changing time_transformation_size with time transformation cuts is
    // supported by the pipe, but some modifications have to be done in
    // SliceWindow, OpenGl buffers.
    ui.timeTransformationSizeSpinBox->setEnabled(!is_raw && !cd_.time_transformation_cuts_enabled);
    ui.timeTransformationSizeSpinBox->setValue(cd_.time_transformation_size);
    ui.TimeTransformationCutsCheckBox->setEnabled(ui.timeTransformationSizeSpinBox->value() >=
                                                  MIN_IMG_NB_TIME_TRANSFORMATION_CUTS);

    ui.WaveLengthDoubleSpinBox->setEnabled(!is_raw);
    ui.WaveLengthDoubleSpinBox->setValue(cd_.lambda * 1.0e9f);
    ui.ZDoubleSpinBox->setEnabled(!is_raw);
    ui.ZDoubleSpinBox->setValue(cd_.zdistance);
    ui.BoundaryLineEdit->setText(QString::number(holovibes_.get_boundary()));

    // Filter2d
    ui.Filter2D->setEnabled(!is_raw);
    ui.Filter2D->setChecked(!is_raw && cd_.filter2d_enabled);
    ui.Filter2DView->setEnabled(!is_raw && cd_.filter2d_enabled);
    ui.Filter2DView->setChecked(!is_raw && cd_.filter2d_view_enabled);
    ui.Filter2DN1SpinBox->setEnabled(!is_raw && cd_.filter2d_enabled);
    ui.Filter2DN1SpinBox->setValue(cd_.filter2d_n1);
    ui.Filter2DN1SpinBox->setMaximum(ui.Filter2DN2SpinBox->value() - 1);
    ui.Filter2DN2SpinBox->setEnabled(!is_raw && cd_.filter2d_enabled);
    ui.Filter2DN2SpinBox->setValue(cd_.filter2d_n2);

    // Composite
    const int time_transformation_size_max = cd_.time_transformation_size - 1;
    ui.PRedSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui.PBlueSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui.SpinBox_hue_freq_min->setMaximum(time_transformation_size_max);
    ui.SpinBox_hue_freq_max->setMaximum(time_transformation_size_max);
    ui.SpinBox_saturation_freq_min->setMaximum(time_transformation_size_max);
    ui.SpinBox_saturation_freq_max->setMaximum(time_transformation_size_max);
    ui.SpinBox_value_freq_min->setMaximum(time_transformation_size_max);
    ui.SpinBox_value_freq_max->setMaximum(time_transformation_size_max);

    ui.RenormalizationCheckBox->setChecked(cd_.composite_auto_weights_);

    QSpinBoxQuietSetValue(ui.PRedSpinBox_Composite, cd_.composite_p_red);
    QSpinBoxQuietSetValue(ui.PBlueSpinBox_Composite, cd_.composite_p_blue);
    QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_R, cd_.weight_r);
    QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_G, cd_.weight_g);
    QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_B, cd_.weight_b);
    ui.CompositePanel->actualize_frequency_channel_v();

    QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_min, cd_.composite_p_min_h);
    QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_max, cd_.composite_p_max_h);
    QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_min, (int)(cd_.slider_h_threshold_min * 1000));
    ui.CompositePanel->slide_update_threshold_h_min();
    QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_max, (int)(cd_.slider_h_threshold_max * 1000));
    ui.CompositePanel->slide_update_threshold_h_max();

    QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_min, cd_.composite_p_min_s);
    QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_max, cd_.composite_p_max_s);
    QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_min, (int)(cd_.slider_s_threshold_min * 1000));
    ui.CompositePanel->slide_update_threshold_s_min();
    QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_max, (int)(cd_.slider_s_threshold_max * 1000));
    ui.CompositePanel->slide_update_threshold_s_max();

    QSpinBoxQuietSetValue(ui.SpinBox_value_freq_min, cd_.composite_p_min_v);
    QSpinBoxQuietSetValue(ui.SpinBox_value_freq_max, cd_.composite_p_max_v);
    QSliderQuietSetValue(ui.horizontalSlider_value_threshold_min, (int)(cd_.slider_v_threshold_min * 1000));
    ui.CompositePanel->slide_update_threshold_v_min();
    QSliderQuietSetValue(ui.horizontalSlider_value_threshold_max, (int)(cd_.slider_v_threshold_max * 1000));
    ui.CompositePanel->slide_update_threshold_v_max();

    ui.CompositePanel->setHidden(is_raw_mode() || (cd_.img_type != ImgType::Composite));

    bool rgbMode = ui.radioButton_rgb->isChecked();
    ui.groupBox->setHidden(!rgbMode);
    ui.groupBox_5->setHidden(!rgbMode && !ui.RenormalizationCheckBox->isChecked());
    ui.groupBox_hue->setHidden(rgbMode);
    ui.groupBox_saturation->setHidden(rgbMode);
    ui.groupBox_value->setHidden(rgbMode);

    // Reticle
    ui.ReticleScaleDoubleSpinBox->setEnabled(cd_.reticle_enabled);
    ui.ReticleScaleDoubleSpinBox->setValue(cd_.reticle_scale);
    ui.DisplayReticleCheckBox->setChecked(cd_.reticle_enabled);

    // Lens View
    ui.LensViewCheckBox->setChecked(cd_.gpu_lens_display_enabled);

    // Renormalize
    ui.RenormalizeCheckBox->setChecked(cd_.renorm_enabled);

    // Convolution
    ui.ConvoCheckBox->setEnabled(cd_.compute_mode == Computation::Hologram);
    ui.ConvoCheckBox->setChecked(cd_.convolution_enabled);
    ui.DivideConvoCheckBox->setChecked(cd_.convolution_enabled && cd_.divide_convolution_enabled);

    QLineEdit* path_line_edit = ui.OutputFilePathLineEdit;
    path_line_edit->clear();

    std::string record_output_path =
        (std::filesystem::path(record_output_directory_) / default_output_filename_).string();
    path_line_edit->insert(record_output_path.c_str());
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
                                                    file_input_directory_.c_str(),
                                                    tr("All files (*.ini);; Ini files (*.ini)"));

    reload_ini(filename);
}

void MainWindow::reload_ini() { reload_ini(""); }

void MainWindow::reload_ini(QString filename)
{
    ui.ImportPanel->import_stop();
    try
    {
        load_ini(filename.isEmpty() ? ::holovibes::ini::get_global_ini_path() : filename.toStdString());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        LOG_INFO << e.what() << std::endl;
    }

    if (import_type_ == ImportType::File)
        ui.ImportPanel->import_start();
    else if (import_type_ == ImportType::Camera)
    {
        change_camera(kCamera);
    }
    notify();
}

void MainWindow::load_ini(const std::string& path)
{
    boost::property_tree::ptree ptree;
    GroupBox* image_rendering_group_box = ui.ImageRenderingGroupBox;
    Panel* view_panel = ui.ViewPanel;
    Panel* import_panel = ui.ImportPanel;
    Panel* info_panel = ui.InfoPanel;

    QAction* image_rendering_action = ui.actionImage_rendering;
    QAction* view_action = ui.actionView;
    QAction* import_export_action = ui.actionImportExport;
    QAction* info_action = ui.actionInfo;

    boost::property_tree::ini_parser::read_ini(path, ptree);

    if (!ptree.empty())
    {
        // Load general compute data
        ini::load_ini(ptree, cd_);

        // Load window specific data
        default_output_filename_ = ptree.get<std::string>("files.default_output_filename", default_output_filename_);
        record_output_directory_ = ptree.get<std::string>("files.record_output_directory", record_output_directory_);
        file_input_directory_ = ptree.get<std::string>("files.file_input_directory", file_input_directory_);
        batch_input_directory_ = ptree.get<std::string>("files.batch_input_directory", batch_input_directory_);

        image_rendering_action->setChecked(
            !ptree.get<bool>("image_rendering.hidden", image_rendering_group_box->isHidden()));

        const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
        if (z_step > 0.0f)
            set_z_step(z_step);

        view_action->setChecked(!ptree.get<bool>("view.hidden", view_panel->isHidden()));

        last_img_type_ = cd_.img_type == ImgType::Composite ? "Composite image" : last_img_type_;

        ui.ViewModeComboBox->setCurrentIndex(static_cast<int>(cd_.img_type.load()));

        displayAngle = ptree.get("view.mainWindow_rotate", displayAngle);
        xzAngle = ptree.get<float>("view.xCut_rotate", xzAngle);
        yzAngle = ptree.get<float>("view.yCut_rotate", yzAngle);
        displayFlip = ptree.get("view.mainWindow_flip", displayFlip);
        xzFlip = ptree.get("view.xCut_flip", xzFlip);
        yzFlip = ptree.get("view.yCut_flip", yzFlip);

        auto_scale_point_threshold_ =
            ptree.get<size_t>("chart.auto_scale_point_threshold", auto_scale_point_threshold_);

        const uint record_frame_step = ptree.get<uint>("record.record_frame_step", record_frame_step_);
        ui.ExportPanel->set_record_frame_step(record_frame_step);

        import_export_action->setChecked(!ptree.get<bool>("import_export.hidden", import_panel->isHidden()));

        ui.ImportInputFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));

        info_action->setChecked(!ptree.get<bool>("info.hidden", info_panel->isHidden()));
        theme_index_ = ptree.get<int>("info.theme_type", theme_index_);

        window_max_size = ptree.get<uint>("display.main_window_max_size", 768);
        time_transformation_cuts_window_max_size =
            ptree.get<uint>("display.time_transformation_cuts_window_max_size", 512);
        auxiliary_window_max_size = ptree.get<uint>("display.auxiliary_window_max_size", 512);

        notify();
    }
}

void MainWindow::save_ini(const std::string& path)
{
    boost::property_tree::ptree ptree;
    GroupBox* image_rendering_group_box = ui.ImageRenderingGroupBox;
    Panel* view_panel = ui.ViewPanel;
    Frame* import_export_frame = ui.ImportExportFrame;
    Panel* info_panel = ui.InfoPanel;
    Config& config = global::global_config;

    // Save general compute data
    ini::save_ini(ptree, cd_);

    // Save window specific data
    ptree.put<std::string>("files.default_output_filename", default_output_filename_);
    ptree.put<std::string>("files.record_output_directory", record_output_directory_);
    ptree.put<std::string>("files.file_input_directory", file_input_directory_);
    ptree.put<std::string>("files.batch_input_directory", batch_input_directory_);

    ptree.put<bool>("image_rendering.hidden", image_rendering_group_box->isHidden());

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

    ptree.put<uint>("record.record_frame_step", record_frame_step_);

    ptree.put<bool>("import_export.hidden", import_export_frame->isHidden());

    ptree.put<bool>("info.hidden", info_panel->isHidden());
    ptree.put<ushort>("info.theme_type", theme_index_);

    ptree.put<uint>("display.main_window_max_size", window_max_size);
    ptree.put<uint>("display.time_transformation_cuts_window_max_size", time_transformation_cuts_window_max_size);
    ptree.put<uint>("display.auxiliary_window_max_size", auxiliary_window_max_size);

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
        set_convolution_mode(false);

    if (cd_.time_transformation_cuts_enabled)
        ui.ViewPanel->cancel_time_transformation_cuts();

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
    ui.actionSettings->setEnabled(false);
    is_enabled_camera_ = false;

    cd_.set_computation_stopped(true);
    notify();
}

void MainWindow::remove_infos() { Holovibes::instance().get_info_container().clear(); }

void MainWindow::close_windows()
{
    sliceXZ.reset(nullptr);
    sliceYZ.reset(nullptr);

    plot_window_.reset(nullptr);
    mainDisplay.reset(nullptr);

    lens_window.reset(nullptr);
    filter2d_window.reset(nullptr);

    /* Raw view & recording */
    raw_window.reset(nullptr);

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

            set_computation_mode();

            holovibes_.start_camera_frame_read(c);
            is_enabled_camera_ = true;
            set_image_mode(nullptr);
            import_type_ = ImportType::Camera;
            kCamera = c;

            // Make camera's settings menu accessible
            QAction* settings = ui.actionSettings;
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

void MainWindow::set_raw_mode()
{
    close_windows();
    close_critical_compute();

    if (is_enabled_camera_)
    {
        QPoint pos(0, 0);
        const FrameDescriptor& fd = holovibes_.get_gpu_input_queue()->get_fd();
        unsigned short width = fd.width;
        unsigned short height = fd.height;

        get_good_size(width, height, window_max_size);
        QSize size(width, height);
        init_image_mode(pos, size);
        cd_.set_compute_mode(Computation::Raw);
        createPipe();

        mainDisplay.reset(new RawWindow(pos, size, holovibes_.get_gpu_input_queue().get()));
        mainDisplay->setTitle(QString("XY view"));
        mainDisplay->setCd(&cd_);
        mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));

        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        Holovibes::instance().get_info_container().add_indication(InformationContainer::IndicationType::INPUT_FORMAT,
                                                                  fd_info);
        set_convolution_mode(false);
        set_divide_convolution_mode(false);
        notify();
        layout_toggled();
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
                                         sliceXZ,
                                         sliceYZ,
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

void MainWindow::set_holographic_mode()
{
    // That function is used to reallocate the buffers since the Square
    // input mode could have changed
    /* Close windows & destory thread compute */
    close_windows();
    close_critical_compute();

    /* ---------- */
    try
    {
        cd_.set_compute_mode(Computation::Hologram);
        /* Pipe & Window */
        createPipe();
        createHoloWindow();
        /* Info Manager */
        const FrameDescriptor& fd = holovibes_.get_gpu_output_queue()->get_fd();
        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        Holovibes::instance().get_info_container().add_indication(InformationContainer::IndicationType::OUTPUT_FORMAT,
                                                                  fd_info);
        /* Contrast */
        cd_.set_contrast_enabled(true);

        /* Filter2D */
        ui.Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));

        /* Record Frame Calculation */
        ui.NumberOfFramesSpinBox->setValue(
            ceil((ui.ImportEndIndexSpinBox->value() - ui.ImportStartIndexSpinBox->value()) /
                 (float)ui.TimeTransformationStrideSpinBox->value()));

        /* Notify */
        notify();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot set holographic mode: " << e.what();
    }
}

void MainWindow::set_computation_mode()
{
    if (ui.ImageModeComboBox->currentIndex() == 0)
    {
        cd_.set_compute_mode(Computation::Raw);
    }
    else if (ui.ImageModeComboBox->currentIndex() == 1)
    {
        cd_.set_compute_mode(Computation::Hologram);
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
    cd_.img_type = static_cast<ImgType>(ui.ViewModeComboBox->currentIndex());
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

            ui.PRedSpinBox_Composite->setValue(min_val_composite);
            ui.SpinBox_hue_freq_min->setValue(min_val_composite);
            ui.SpinBox_saturation_freq_min->setValue(min_val_composite);
            ui.SpinBox_value_freq_min->setValue(min_val_composite);

            ui.PBlueSpinBox_Composite->setValue(max_val_composite);
            ui.SpinBox_hue_freq_max->setValue(max_val_composite);
            ui.SpinBox_saturation_freq_max->setValue(max_val_composite);
            ui.SpinBox_value_freq_max->setValue(max_val_composite);
        }
    }
    last_img_type_ = value;

    auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());

    pipe->insert_fn_end_vect([=]() {
        cd_.set_img_type(static_cast<ImgType>(ui.ViewModeComboBox->currentIndex()));
        notify();
        layout_toggled();
    });
    pipe_refresh();

    // Force XYview autocontrast
    pipe->autocontrast_end_pipe(WindowKind::XYview);
    // Force cuts views autocontrast if needed
    if (cd_.time_transformation_cuts_enabled)
        ui.ViewPanel->set_auto_contrast_cuts();
}

bool MainWindow::is_raw_mode() { return cd_.compute_mode == Computation::Raw; }

void MainWindow::set_image_mode(QString mode)
{
    if (mode != nullptr)
    {
        // Call comes from ui
        if (ui.ImageModeComboBox->currentIndex() == 0)
            set_raw_mode();
        else
            set_holographic_mode();
    }
    else if (cd_.compute_mode == Computation::Raw)
        set_raw_mode();
    else if (cd_.compute_mode == Computation::Hologram)
        set_holographic_mode();
}
#pragma endregion

#pragma region Batch
void MainWindow::update_batch_size()
{
    if (is_raw_mode())
        return;

    int value = ui.BatchSizeSpinBox->value();

    if (value == cd_.batch_size)
        return;

    auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            cd_.set_batch_size(value);
            cd_.adapt_time_transformation_stride();
            holovibes_.get_compute_pipe()->request_update_batch_size();
            notify();
        });
    }
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

#pragma endregion
/* ------------ */
#pragma region STFT
void MainWindow::update_time_transformation_stride()
{
    if (is_raw_mode())
        return;

    int value = ui.TimeTransformationStrideSpinBox->value();

    if (value == cd_.time_transformation_stride)
        return;

    auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            cd_.set_time_transformation_stride(value);
            cd_.adapt_time_transformation_stride();
            holovibes_.get_compute_pipe()->request_update_time_transformation_stride();
            ui.NumberOfFramesSpinBox->setValue(
                ceil((ui.ImportEndIndexSpinBox->value() - ui.ImportStartIndexSpinBox->value()) /
                     (float)ui.TimeTransformationStrideSpinBox->value()));
            notify();
        });
    }
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}
#pragma endregion
/* ------------ */
#pragma region Computation
void MainWindow::change_window()
{
    QComboBox* window_cbox = ui.WindowSelectionComboBox;

    cd_.change_window(window_cbox->currentIndex());
    pipe_refresh();
    notify();
}

void MainWindow::set_filter2d(bool checked)
{
    if (is_raw_mode())
        return;

    if (!checked)
    {
        cd_.set_filter2d_enabled(checked);
        cancel_filter2d();
    }
    else
    {
        const camera::FrameDescriptor& fd = holovibes_.get_gpu_input_queue()->get_fd();

        // Set the input box related to the filter2d
        ui.Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));
        set_filter2d_n2(ui.Filter2DN2SpinBox->value());
        set_filter2d_n1(ui.Filter2DN1SpinBox->value());

        if (auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get()))
            pipe->autocontrast_end_pipe(WindowKind::XYview);
        cd_.set_filter2d_enabled(checked);
    }
    pipe_refresh();
    notify();
}

void MainWindow::disable_filter2d_view()
{

    auto pipe = holovibes_.get_compute_pipe();
    pipe->request_disable_filter2d_view();

    // Wait for the filter2d view to be disabled for notify
    while (pipe->get_disable_filter2d_view_requested())
        continue;

    if (filter2d_window)
    {
        // Remove the on triggered event

        disconnect(filter2d_window.get(), SIGNAL(destroyed()), this, SLOT(disable_filter2d_view()));
    }

    // Change the focused window
    change_window();

    notify();
}

void MainWindow::update_filter2d_view(bool checked)
{
    if (is_raw_mode())
        return;

    if (checked)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 310, 0);
            auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());
            if (pipe)
            {
                pipe->request_filter2d_view();

                const FrameDescriptor& fd = holovibes_.get_gpu_output_queue()->get_fd();
                ushort filter2d_window_width = fd.width;
                ushort filter2d_window_height = fd.height;
                get_good_size(filter2d_window_width, filter2d_window_height, auxiliary_window_max_size);

                // Wait for the filter2d view to be enabled for notify
                while (pipe->get_filter2d_view_requested())
                    continue;

                filter2d_window.reset(new Filter2DWindow(pos,
                                                         QSize(filter2d_window_width, filter2d_window_height),
                                                         pipe->get_filter2d_view_queue().get(),
                                                         this));

                filter2d_window->setTitle("Filter2D view");
                filter2d_window->setCd(&cd_);

                connect(filter2d_window.get(), SIGNAL(destroyed()), this, SLOT(disable_filter2d_view()));
                cd_.set_log_scale_slice_enabled(WindowKind::Filter2D, true);
                pipe->autocontrast_end_pipe(WindowKind::Filter2D);
            }
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what() << std::endl;
        }
    }

    else
    {
        disable_filter2d_view();
        filter2d_window.reset(nullptr);
    }

    pipe_refresh();
    notify();
}

void MainWindow::set_filter2d_pipe()
{
    if (auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XYview);
        if (cd_.time_transformation_cuts_enabled)
        {
            pipe->autocontrast_end_pipe(WindowKind::XZview);
            pipe->autocontrast_end_pipe(WindowKind::YZview);
        }
        if (cd_.filter2d_view_enabled)
            pipe->autocontrast_end_pipe(WindowKind::Filter2D);
    }

    pipe_refresh();
    notify();
}

void MainWindow::set_filter2d_n1(int n)
{
    if (is_raw_mode())
        return;

    cd_.set_filter2d_n1(n);
    set_filter2d_pipe();
}

void MainWindow::set_filter2d_n2(int n)
{
    if (is_raw_mode())
        return;

    cd_.set_filter2d_n2(n);
    set_filter2d_pipe();
}

void MainWindow::cancel_filter2d()
{
    if (is_raw_mode())
        return;

    if (cd_.filter2d_view_enabled)
        update_filter2d_view(false);
    pipe_refresh();
    notify();
}

void MainWindow::set_time_transformation_size()
{
    if (is_raw_mode())
        return;

    int time_transformation_size = ui.timeTransformationSizeSpinBox->value();
    time_transformation_size = std::max(1, time_transformation_size);

    if (time_transformation_size == cd_.time_transformation_size)
        return;
    notify();
    auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            cd_.set_time_transformation_size(time_transformation_size);
            holovibes_.get_compute_pipe()->request_update_time_transformation_size();
            ui.ViewPanel->set_p_accu();
            // This will not do anything until
            // SliceWindow::changeTexture() isn't coded.
        });
    }
}

void MainWindow::set_wavelength(const double value)
{
    if (is_raw_mode())
        return;

    cd_.set_lambda(static_cast<float>(value) * 1.0e-9f);
    pipe_refresh();
}

void MainWindow::set_z(const double value)
{
    if (is_raw_mode())
        return;

    cd_.set_zdistance(static_cast<float>(value));
    pipe_refresh();
}

void MainWindow::increment_z()
{
    if (is_raw_mode())
        return;

    set_z(cd_.zdistance + z_step_);
    ui.ZDoubleSpinBox->setValue(cd_.zdistance);
}

void MainWindow::decrement_z()
{
    if (is_raw_mode())
        return;

    set_z(cd_.zdistance - z_step_);
    ui.ZDoubleSpinBox->setValue(cd_.zdistance);
}

void MainWindow::set_z_step(const double value)
{
    z_step_ = value;
    ui.ZDoubleSpinBox->setSingleStep(value);
}

void MainWindow::set_space_transformation(const QString& value)
{
    if (is_raw_mode())
        return;

    cd_.set_space_transformation_from_string(value.toStdString());
    set_holographic_mode();
}

void MainWindow::set_time_transformation(const QString& value)
{
    if (is_raw_mode())
        return;

    cd_.set_time_transformation_from_string(value.toStdString());
    set_holographic_mode();
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
#pragma region Contrast - Log
void MainWindow::QSpinBoxQuietSetValue(QSpinBox* spinBox, int value)
{
    spinBox->blockSignals(true);
    spinBox->setValue(value);
    spinBox->blockSignals(false);
}

void MainWindow::QSliderQuietSetValue(QSlider* slider, int value)
{
    slider->blockSignals(true);
    slider->setValue(value);
    slider->blockSignals(false);
}

void MainWindow::QDoubleSpinBoxQuietSetValue(QDoubleSpinBox* spinBox, double value)
{
    spinBox->blockSignals(true);
    spinBox->setValue(value);
    spinBox->blockSignals(false);
}
#pragma endregion
/* ------------ */
#pragma region Convolution
void MainWindow::update_convo_kernel(const QString& value)
{
    if (cd_.convolution_enabled)
    {
        cd_.set_convolution(true, ui.KernelQuickSelectComboBox->currentText().toStdString());

        try
        {
            auto pipe = holovibes_.get_compute_pipe();
            pipe->request_convolution();
            // Wait for the convolution to be enabled for notify
            while (pipe->get_convolution_requested())
                continue;
        }
        catch (const std::exception& e)
        {
            cd_.set_convolution_enabled(false);
            LOG_ERROR << e.what();
        }

        notify();
    }
}

void MainWindow::set_convolution_mode(const bool value)
{
    cd_.set_convolution(value, ui.KernelQuickSelectComboBox->currentText().toStdString());

    try
    {
        auto pipe = holovibes_.get_compute_pipe();

        if (value)
        {
            pipe->request_convolution();
            // Wait for the convolution to be enabled for notify
            while (pipe->get_convolution_requested())
                continue;
        }
        else
        {
            pipe->request_disable_convolution();
            // Wait for the convolution to be disabled for notify
            while (pipe->get_disable_convolution_requested())
                continue;
        }
    }
    catch (const std::exception& e)
    {
        cd_.set_convolution_enabled(false);
        LOG_ERROR << e.what();
    }

    notify();
}

void MainWindow::set_divide_convolution_mode(const bool value)
{
    cd_.set_divide_convolution_mode(value);

    pipe_refresh();
    notify();
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

#pragma region Getters

RawWindow* MainWindow::get_main_display() { return mainDisplay.get(); }

void MainWindow::update_file_reader_index(int n)
{
    auto lambda = [this, n]() { ui.InfoPanel->update_file_reader_progress(n); };
    synchronize_thread(lambda);
}
#pragma endregion
} // namespace gui
} // namespace holovibes
