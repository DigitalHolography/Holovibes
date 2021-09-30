#include <filesystem>
#include <algorithm>
#include <list>
#include <optional>
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
#include "API.hh"

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
{
    ui.setupUi(this);

    qRegisterMetaType<std::function<void()>>();
    connect(this,
            SIGNAL(synchronize_thread_signal(std::function<void()>)),
            this,
            SLOT(synchronize_thread(std::function<void()>)));

    setWindowIcon(QIcon("Holovibes.ico"));

    auto display_info_text_fun = [=](const std::string& text) {
        synchronize_thread([=]() { ui.InfoTextEdit->setText(text.c_str()); });
    };
    Holovibes::instance().get_info_container().set_display_info_text_function(display_info_text_fun);

    auto update_progress = [=](InformationContainer::ProgressType type, const size_t value, const size_t max_size) {
        synchronize_thread([=]() {
            switch (type)
            {
            case InformationContainer::ProgressType::FILE_READ:
                ui.FileReaderProgressBar->setMaximum(static_cast<int>(max_size));
                ui.FileReaderProgressBar->setValue(static_cast<int>(value));
                break;
            case InformationContainer::ProgressType::CHART_RECORD:
            case InformationContainer::ProgressType::FRAME_RECORD:
                ui.RecordProgressBar->setMaximum(static_cast<int>(max_size));
                ui.RecordProgressBar->setValue(static_cast<int>(value));
                break;
            default:
                return;
            };
        });
    };
    Holovibes::instance().get_info_container().set_update_progress_function(update_progress);
    ui.FileReaderProgressBar->hide();
    ui.RecordProgressBar->hide();

    set_record_mode(QString::fromUtf8("Raw Image"));

    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();

    // need the correct dimensions of main windows
    move(QPoint((screen_width - 800) / 2, (screen_height - 500) / 2));

    // Hide non default tab
    ui.CompositeGroupBox->hide();

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

    set_z_step(ui_descriptor_.z_step_);
    set_record_frame_step(ui_descriptor_.record_frame_step_);
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
    connect(p_left_shortcut_, SIGNAL(activated()), this, SLOT(decrement_p()));

    p_right_shortcut_ = new QShortcut(QKeySequence("Right"), this);
    p_right_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(p_right_shortcut_, SIGNAL(activated()), this, SLOT(increment_p()));

    QComboBox* window_cbox = ui.WindowSelectionComboBox;
    connect(window_cbox, SIGNAL(currentIndexChanged(QString)), this, SLOT(change_window()));

    // Display default values
    ui_descriptor_.holovibes_.get_cd().compute_mode = Computation::Raw;
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
    LOG_INFO;
    delete z_up_shortcut_;
    delete z_down_shortcut_;
    delete p_left_shortcut_;
    delete p_right_shortcut_;

    ::holovibes::api::close_windows(ui_descriptor_.holovibes_,
                                    ui_descriptor_.mainDisplay,
                                    ui_descriptor_.sliceXZ,
                                    ui_descriptor_.sliceYZ,
                                    ui_descriptor_.lens_window,
                                    ui_descriptor_.raw_window,
                                    ui_descriptor_.filter2d_window,
                                    ui_descriptor_.plot_window_);
    ::holovibes::api::close_critical_compute(ui_descriptor_);
    camera_none();
    ::holovibes::api::remove_infos();

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
    LOG_INFO;
    synchronize_thread([this]() { on_notify(); });
}

void MainWindow::on_notify()
{
    LOG_INFO;
    ui.InputBrowseToolButton->setEnabled(ui_descriptor_.holovibes_.get_cd().is_computation_stopped);

    // Tabs
    if (ui_descriptor_.holovibes_.get_cd().is_computation_stopped)
    {
        ui.CompositeGroupBox->hide();
        ui.ImageRenderingGroupBox->setEnabled(false);
        ui.ViewGroupBox->setEnabled(false);
        ui.ExportGroupBox->setEnabled(false);
        layout_toggled();
        return;
    }

    if (ui_descriptor_.is_enabled_camera_ && ui_descriptor_.holovibes_.get_cd().compute_mode == Computation::Raw)
    {
        ui.ImageRenderingGroupBox->setEnabled(true);
        ui.ViewGroupBox->setEnabled(false);
        ui.ExportGroupBox->setEnabled(true);
    }

    else if (ui_descriptor_.is_enabled_camera_ &&
             ui_descriptor_.holovibes_.get_cd().compute_mode == Computation::Hologram)
    {
        ui.ImageRenderingGroupBox->setEnabled(true);
        ui.ViewGroupBox->setEnabled(true);
        ui.ExportGroupBox->setEnabled(true);
    }

    const bool is_raw = ::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_);

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
    ui.RawDisplayingCheckBox->setChecked(!is_raw && ui_descriptor_.holovibes_.get_cd().raw_view_enabled);

    QPushButton* signalBtn = ui.ChartSignalPushButton;
    signalBtn->setStyleSheet((ui_descriptor_.mainDisplay && signalBtn->isEnabled() &&
                              ui_descriptor_.mainDisplay->getKindOfOverlay() == KindOfOverlay::Signal)
                                 ? "QPushButton {color: #8E66D9;}"
                                 : "");

    QPushButton* noiseBtn = ui.ChartNoisePushButton;
    noiseBtn->setStyleSheet((ui_descriptor_.mainDisplay && noiseBtn->isEnabled() &&
                             ui_descriptor_.mainDisplay->getKindOfOverlay() == KindOfOverlay::Noise)
                                ? "QPushButton {color: #00A4AB;}"
                                : "");

    ui.PhaseUnwrap2DCheckBox->setEnabled(ui_descriptor_.holovibes_.get_cd().img_type == ImgType::PhaseIncrease ||
                                         ui_descriptor_.holovibes_.get_cd().img_type == ImgType::Argument);

    // Time transformation cuts
    ui.TimeTransformationCutsCheckBox->setChecked(!is_raw &&
                                                  ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled);

    // Contrast
    ui.ContrastCheckBox->setChecked(!is_raw && ui_descriptor_.holovibes_.get_cd().contrast_enabled);
    ui.ContrastCheckBox->setEnabled(true);
    ui.AutoRefreshContrastCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().contrast_auto_refresh);

    // Contrast SpinBox:
    ui.ContrastMinDoubleSpinBox->setEnabled(!ui_descriptor_.holovibes_.get_cd().contrast_auto_refresh);
    ui.ContrastMinDoubleSpinBox->setValue(
        ui_descriptor_.holovibes_.get_cd().get_contrast_min(ui_descriptor_.holovibes_.get_cd().current_window));
    ui.ContrastMaxDoubleSpinBox->setEnabled(!ui_descriptor_.holovibes_.get_cd().contrast_auto_refresh);
    ui.ContrastMaxDoubleSpinBox->setValue(
        ui_descriptor_.holovibes_.get_cd().get_contrast_max(ui_descriptor_.holovibes_.get_cd().current_window));

    // FFT shift
    ui.FFTShiftCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().fft_shift_enabled);
    ui.FFTShiftCheckBox->setEnabled(true);

    // Window selection
    QComboBox* window_selection = ui.WindowSelectionComboBox;
    window_selection->setEnabled(ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled);
    window_selection->setCurrentIndex(
        window_selection->isEnabled() ? static_cast<int>(ui_descriptor_.holovibes_.get_cd().current_window.load()) : 0);

    ui.LogScaleCheckBox->setEnabled(true);
    ui.LogScaleCheckBox->setChecked(!is_raw && ui_descriptor_.holovibes_.get_cd().get_img_log_scale_slice_enabled(
                                                   ui_descriptor_.holovibes_.get_cd().current_window.load()));
    ui.ImgAccuCheckBox->setEnabled(true);
    ui.ImgAccuCheckBox->setChecked(!is_raw && ui_descriptor_.holovibes_.get_cd().get_img_acc_slice_enabled(
                                                  ui_descriptor_.holovibes_.get_cd().current_window.load()));
    ui.ImgAccuSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().get_img_acc_slice_level(
        ui_descriptor_.holovibes_.get_cd().current_window.load()));
    if (ui_descriptor_.holovibes_.get_cd().current_window == WindowKind::XYview)
    {
        ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(ui_descriptor_.displayAngle))).c_str());
        ui.FlipPushButton->setText(("Flip " + std::to_string(ui_descriptor_.displayFlip)).c_str());
    }
    else if (ui_descriptor_.holovibes_.get_cd().current_window == WindowKind::XZview)
    {
        ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(ui_descriptor_.xzAngle))).c_str());
        ui.FlipPushButton->setText(("Flip " + std::to_string(ui_descriptor_.xzFlip)).c_str());
    }
    else if (ui_descriptor_.holovibes_.get_cd().current_window == WindowKind::YZview)
    {
        ui.RotatePushButton->setText(("Rot " + std::to_string(static_cast<int>(ui_descriptor_.yzAngle))).c_str());
        ui.FlipPushButton->setText(("Flip " + std::to_string(ui_descriptor_.yzFlip)).c_str());
    }

    // p accu
    ui.PAccuCheckBox->setEnabled(ui_descriptor_.holovibes_.get_cd().img_type != ImgType::PhaseIncrease);
    ui.PAccuCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().p_accu_enabled);
    ui.PAccSpinBox->setMaximum(ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1);
    if (ui_descriptor_.holovibes_.get_cd().p_acc_level >
        ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1)
        ui_descriptor_.holovibes_.get_cd().p_acc_level =
            ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1;
    ui.PAccSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().p_acc_level);
    ui.PAccSpinBox->setEnabled(ui_descriptor_.holovibes_.get_cd().img_type != ImgType::PhaseIncrease);
    if (ui_descriptor_.holovibes_.get_cd().p_accu_enabled)
    {
        ui.PSpinBox->setMaximum(ui_descriptor_.holovibes_.get_cd().time_transformation_size -
                                ui_descriptor_.holovibes_.get_cd().p_acc_level - 1);
        if (ui_descriptor_.holovibes_.get_cd().pindex > ui_descriptor_.holovibes_.get_cd().time_transformation_size -
                                                            ui_descriptor_.holovibes_.get_cd().p_acc_level - 1)
            ui_descriptor_.holovibes_.get_cd().pindex = ui_descriptor_.holovibes_.get_cd().time_transformation_size -
                                                        ui_descriptor_.holovibes_.get_cd().p_acc_level - 1;
        ui.PSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().pindex);
        ui.PAccSpinBox->setMaximum(ui_descriptor_.holovibes_.get_cd().time_transformation_size -
                                   ui_descriptor_.holovibes_.get_cd().pindex - 1);
    }
    else
    {
        ui.PSpinBox->setMaximum(ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1);
        if (ui_descriptor_.holovibes_.get_cd().pindex > ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1)
            ui_descriptor_.holovibes_.get_cd().pindex = ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1;
        ui.PSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().pindex);
    }
    ui.PSpinBox->setEnabled(!is_raw);

    // q accu
    bool is_ssa_stft = ui_descriptor_.holovibes_.get_cd().time_transformation == TimeTransformation::SSA_STFT;
    ui.Q_AccuCheckBox->setEnabled(is_ssa_stft && !is_raw);
    ui.Q_AccSpinBox->setEnabled(is_ssa_stft && !is_raw);
    ui.Q_SpinBox->setEnabled(is_ssa_stft && !is_raw);

    ui.Q_AccuCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().q_acc_enabled);
    ui.Q_AccSpinBox->setMaximum(ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1);
    if (ui_descriptor_.holovibes_.get_cd().q_acc_level >
        ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1)
        ui_descriptor_.holovibes_.get_cd().q_acc_level =
            ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1;
    ui.Q_AccSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().q_acc_level);
    if (ui_descriptor_.holovibes_.get_cd().q_acc_enabled)
    {
        ui.Q_SpinBox->setMaximum(ui_descriptor_.holovibes_.get_cd().time_transformation_size -
                                 ui_descriptor_.holovibes_.get_cd().q_acc_level - 1);
        if (ui_descriptor_.holovibes_.get_cd().q_index > ui_descriptor_.holovibes_.get_cd().time_transformation_size -
                                                             ui_descriptor_.holovibes_.get_cd().q_acc_level - 1)
            ui_descriptor_.holovibes_.get_cd().q_index = ui_descriptor_.holovibes_.get_cd().time_transformation_size -
                                                         ui_descriptor_.holovibes_.get_cd().q_acc_level - 1;
        ui.Q_SpinBox->setValue(ui_descriptor_.holovibes_.get_cd().q_index);
        ui.Q_AccSpinBox->setMaximum(ui_descriptor_.holovibes_.get_cd().time_transformation_size -
                                    ui_descriptor_.holovibes_.get_cd().q_index - 1);
    }
    else
    {
        ui.Q_SpinBox->setMaximum(ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1);
        if (ui_descriptor_.holovibes_.get_cd().q_index >
            ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1)
            ui_descriptor_.holovibes_.get_cd().q_index =
                ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1;
        ui.Q_SpinBox->setValue(ui_descriptor_.holovibes_.get_cd().q_index);
    }

    // XY accu
    ui.XAccuCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().x_accu_enabled);
    ui.XAccSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().x_acc_level);
    ui.YAccuCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().y_accu_enabled);
    ui.YAccSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().y_acc_level);

    int max_width = 0;
    int max_height = 0;
    if (ui_descriptor_.holovibes_.get_gpu_input_queue() != nullptr)
    {
        max_width = ui_descriptor_.holovibes_.get_gpu_input_queue()->get_fd().width - 1;
        max_height = ui_descriptor_.holovibes_.get_gpu_input_queue()->get_fd().height - 1;
    }
    else
    {
        ui_descriptor_.holovibes_.get_cd().x_cuts = 0;
        ui_descriptor_.holovibes_.get_cd().y_cuts = 0;
    }
    ui.XSpinBox->setMaximum(max_width);
    ui.YSpinBox->setMaximum(max_height);
    QSpinBoxQuietSetValue(ui.XSpinBox, ui_descriptor_.holovibes_.get_cd().x_cuts);
    QSpinBoxQuietSetValue(ui.YSpinBox, ui_descriptor_.holovibes_.get_cd().y_cuts);

    // Time transformation
    ui.TimeTransformationStrideSpinBox->setEnabled(!is_raw && !ui_descriptor_.holovibes_.get_cd().fast_pipe);

    const uint input_queue_capacity = global::global_config.input_queue_max_size;

    ui.TimeTransformationStrideSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().time_transformation_stride);
    ui.TimeTransformationStrideSpinBox->setSingleStep(ui_descriptor_.holovibes_.get_cd().batch_size);
    ui.TimeTransformationStrideSpinBox->setMinimum(ui_descriptor_.holovibes_.get_cd().batch_size);

    // Batch
    ui.BatchSizeSpinBox->setEnabled(!is_raw && !ui_descriptor_.is_recording_ &&
                                    !ui_descriptor_.holovibes_.get_cd().fast_pipe);

    if (ui_descriptor_.holovibes_.get_cd().batch_size > input_queue_capacity)
        ui_descriptor_.holovibes_.get_cd().batch_size = input_queue_capacity;

    ui.BatchSizeSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().batch_size);
    ui.BatchSizeSpinBox->setMaximum(input_queue_capacity);

    // Image rendering
    ui.SpaceTransformationComboBox->setEnabled(!is_raw &&
                                               !ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled);
    ui.SpaceTransformationComboBox->setCurrentIndex(
        static_cast<int>(ui_descriptor_.holovibes_.get_cd().space_transformation.load()));
    ui.TimeTransformationComboBox->setEnabled(!is_raw);
    ui.TimeTransformationComboBox->setCurrentIndex(
        static_cast<int>(ui_descriptor_.holovibes_.get_cd().time_transformation.load()));

    // Changing time_transformation_size with time transformation cuts is
    // supported by the pipe, but some modifications have to be done in
    // SliceWindow, OpenGl buffers.
    ui.timeTransformationSizeSpinBox->setEnabled(!is_raw && !ui_descriptor_.holovibes_.get_cd().fast_pipe &&
                                                 !ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled);
    ui.timeTransformationSizeSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().time_transformation_size);
    ui.TimeTransformationCutsCheckBox->setEnabled(ui.timeTransformationSizeSpinBox->value() >=
                                                  MIN_IMG_NB_TIME_TRANSFORMATION_CUTS);

    ui.WaveLengthDoubleSpinBox->setEnabled(!is_raw);
    ui.WaveLengthDoubleSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().lambda * 1.0e9f);
    ui.ZDoubleSpinBox->setEnabled(!is_raw);
    ui.ZDoubleSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().zdistance);
    ui.BoundaryLineEdit->setText(QString::number(ui_descriptor_.holovibes_.get_boundary()));

    // Filter2d
    ui.Filter2D->setEnabled(!is_raw);
    ui.Filter2D->setChecked(!is_raw && ui_descriptor_.holovibes_.get_cd().filter2d_enabled);
    ui.Filter2DView->setEnabled(!is_raw && ui_descriptor_.holovibes_.get_cd().filter2d_enabled);
    ui.Filter2DView->setChecked(!is_raw && ui_descriptor_.holovibes_.get_cd().filter2d_view_enabled);
    ui.Filter2DN1SpinBox->setEnabled(!is_raw && ui_descriptor_.holovibes_.get_cd().filter2d_enabled);
    ui.Filter2DN1SpinBox->setValue(ui_descriptor_.holovibes_.get_cd().filter2d_n1);
    ui.Filter2DN1SpinBox->setMaximum(ui.Filter2DN2SpinBox->value() - 1);
    ui.Filter2DN2SpinBox->setEnabled(!is_raw && ui_descriptor_.holovibes_.get_cd().filter2d_enabled);
    ui.Filter2DN2SpinBox->setValue(ui_descriptor_.holovibes_.get_cd().filter2d_n2);

    // Composite
    const int time_transformation_size_max = ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1;
    ui.PRedSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui.PBlueSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui.SpinBox_hue_freq_min->setMaximum(time_transformation_size_max);
    ui.SpinBox_hue_freq_max->setMaximum(time_transformation_size_max);
    ui.SpinBox_saturation_freq_min->setMaximum(time_transformation_size_max);
    ui.SpinBox_saturation_freq_max->setMaximum(time_transformation_size_max);
    ui.SpinBox_value_freq_min->setMaximum(time_transformation_size_max);
    ui.SpinBox_value_freq_max->setMaximum(time_transformation_size_max);

    ui.RenormalizationCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().composite_auto_weights_);

    QSpinBoxQuietSetValue(ui.PRedSpinBox_Composite, ui_descriptor_.holovibes_.get_cd().composite_p_red);
    QSpinBoxQuietSetValue(ui.PBlueSpinBox_Composite, ui_descriptor_.holovibes_.get_cd().composite_p_blue);
    QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_R, ui_descriptor_.holovibes_.get_cd().weight_r);
    QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_G, ui_descriptor_.holovibes_.get_cd().weight_g);
    QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_B, ui_descriptor_.holovibes_.get_cd().weight_b);
    actualize_frequency_channel_v();

    QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_min, ui_descriptor_.holovibes_.get_cd().composite_p_min_h);
    QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_max, ui_descriptor_.holovibes_.get_cd().composite_p_max_h);
    QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_min,
                         (int)(ui_descriptor_.holovibes_.get_cd().slider_h_threshold_min * 1000));
    slide_update_threshold_h_min();
    QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_max,
                         (int)(ui_descriptor_.holovibes_.get_cd().slider_h_threshold_max * 1000));
    slide_update_threshold_h_max();

    QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_min, ui_descriptor_.holovibes_.get_cd().composite_p_min_s);
    QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_max, ui_descriptor_.holovibes_.get_cd().composite_p_max_s);
    QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_min,
                         (int)(ui_descriptor_.holovibes_.get_cd().slider_s_threshold_min * 1000));
    slide_update_threshold_s_min();
    QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_max,
                         (int)(ui_descriptor_.holovibes_.get_cd().slider_s_threshold_max * 1000));
    slide_update_threshold_s_max();

    QSpinBoxQuietSetValue(ui.SpinBox_value_freq_min, ui_descriptor_.holovibes_.get_cd().composite_p_min_v);
    QSpinBoxQuietSetValue(ui.SpinBox_value_freq_max, ui_descriptor_.holovibes_.get_cd().composite_p_max_v);
    QSliderQuietSetValue(ui.horizontalSlider_value_threshold_min,
                         (int)(ui_descriptor_.holovibes_.get_cd().slider_v_threshold_min * 1000));
    slide_update_threshold_v_min();
    QSliderQuietSetValue(ui.horizontalSlider_value_threshold_max,
                         (int)(ui_descriptor_.holovibes_.get_cd().slider_v_threshold_max * 1000));
    slide_update_threshold_v_max();

    ui.CompositeGroupBox->setHidden(::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_) ||
                                    (ui_descriptor_.holovibes_.get_cd().img_type != ImgType::Composite));

    bool rgbMode = ui.radioButton_rgb->isChecked();
    ui.groupBox->setHidden(!rgbMode);
    ui.groupBox_5->setHidden(!rgbMode && !ui.RenormalizationCheckBox->isChecked());
    ui.groupBox_hue->setHidden(rgbMode);
    ui.groupBox_saturation->setHidden(rgbMode);
    ui.groupBox_value->setHidden(rgbMode);

    // Reticle
    ui.ReticleScaleDoubleSpinBox->setEnabled(ui_descriptor_.holovibes_.get_cd().reticle_enabled);
    ui.ReticleScaleDoubleSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().reticle_scale);
    ui.DisplayReticleCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().reticle_enabled);

    // Lens View
    ui.LensViewCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().gpu_lens_display_enabled);

    // Renormalize
    ui.RenormalizeCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().renorm_enabled);

    // Convolution
    ui.ConvoCheckBox->setEnabled(ui_descriptor_.holovibes_.get_cd().compute_mode == Computation::Hologram);
    ui.ConvoCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().convolution_enabled);
    ui.DivideConvoCheckBox->setChecked(ui_descriptor_.holovibes_.get_cd().convolution_enabled &&
                                       ui_descriptor_.holovibes_.get_cd().divide_convolution_enabled);

    QLineEdit* path_line_edit = ui.OutputFilePathLineEdit;
    path_line_edit->clear();

    std::string record_output_path =
        (std::filesystem::path(ui_descriptor_.record_output_directory_) / ui_descriptor_.default_output_filename_)
            .string();
    path_line_edit->insert(record_output_path.c_str());
}

void MainWindow::notify_error(const std::exception& e)
{
    LOG_INFO;
    const CustomException* err_ptr = dynamic_cast<const CustomException*>(&e);
    if (err_ptr)
    {
        const UpdateException* err_update_ptr = dynamic_cast<const UpdateException*>(err_ptr);
        if (err_update_ptr)
        {
            auto lambda = [this] {
                // notify will be in close_critical_compute
                ui_descriptor_.holovibes_.get_cd().pindex = 0;
                ui_descriptor_.holovibes_.get_cd().time_transformation_size = 1;
                if (ui_descriptor_.holovibes_.get_cd().convolution_enabled)
                {
                    ui_descriptor_.holovibes_.get_cd().convolution_enabled = false;
                }
                ::holovibes::api::close_windows(ui_descriptor_.holovibes_,
                                                ui_descriptor_.mainDisplay,
                                                ui_descriptor_.sliceXZ,
                                                ui_descriptor_.sliceYZ,
                                                ui_descriptor_.lens_window,
                                                ui_descriptor_.raw_window,
                                                ui_descriptor_.filter2d_window,
                                                ui_descriptor_.plot_window_);
                ::holovibes::api::close_critical_compute(ui_descriptor_);
                LOG_ERROR << "GPU computing error occured.";
                notify();
            };
            synchronize_thread(lambda);
        }

        auto lambda = [this, accu = (dynamic_cast<const AccumulationException*>(err_ptr) != nullptr)] {
            if (accu)
            {
                ui_descriptor_.holovibes_.get_cd().img_acc_slice_xy_enabled = false;
                ui_descriptor_.holovibes_.get_cd().img_acc_slice_xy_level = 1;
            }
            ::holovibes::api::close_critical_compute(ui_descriptor_);

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
    LOG_INFO;

    synchronize_thread([=]() {
        // Resizing to original size, then adjust it to fit the groupboxes
        resize(baseSize());
        adjustSize();
    });
}

void MainWindow::credits()
{
    LOG_INFO;

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
    LOG_INFO;
    QDesktopServices::openUrl(QUrl("https://ftp.espci.fr/incoming/Atlan/holovibes/manual/"));
}

#pragma endregion
/* ------------ */
#pragma region Ini

void MainWindow::configure_holovibes()
{
    LOG_INFO;
    open_file(::holovibes::ini::get_global_ini_path());
}

void MainWindow::write_ini()
{
    LOG_INFO;
    // Saves the current state of holovibes in holovibes.ini located in Holovibes.exe directory
    save_ini(::holovibes::ini::get_global_ini_path());
    notify();
}

void MainWindow::reload_ini()
{
    LOG_INFO;
    import_stop();
    try
    {
        load_ini(::holovibes::ini::get_global_ini_path());
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
        LOG_INFO << e.what() << std::endl;
    }
    if (ui_descriptor_.import_type_ == ::holovibes::UserInterfaceDescriptor::ImportType::File)
    {
        import_start();
    }
    else if (ui_descriptor_.import_type_ == ::holovibes::UserInterfaceDescriptor::ImportType::Camera)
    {
        change_camera(ui_descriptor_.kCamera);
    }
    notify();
}

void MainWindow::load_ini(const std::string& path)
{
    LOG_INFO;
    boost::property_tree::ptree ptree;
    GroupBox* image_rendering_group_box = ui.ImageRenderingGroupBox;
    GroupBox* view_group_box = ui.ViewGroupBox;
    GroupBox* import_group_box = ui.ImportGroupBox;
    GroupBox* info_group_box = ui.InfoGroupBox;

    QAction* image_rendering_action = ui.actionImage_rendering;
    QAction* view_action = ui.actionView;
    QAction* import_export_action = ui.actionImportExport;
    QAction* info_action = ui.actionInfo;

    boost::property_tree::ini_parser::read_ini(path, ptree);

    if (!ptree.empty())
    {
        // Load general compute data
        ini::load_ini(ptree, ui_descriptor_.holovibes_.get_cd());

        // Load window specific data
        ui_descriptor_.default_output_filename_ =
            ptree.get<std::string>("files.default_output_filename", ui_descriptor_.default_output_filename_);
        ui_descriptor_.record_output_directory_ =
            ptree.get<std::string>("files.record_output_directory", ui_descriptor_.record_output_directory_);
        ui_descriptor_.file_input_directory_ =
            ptree.get<std::string>("files.file_input_directory", ui_descriptor_.file_input_directory_);
        ui_descriptor_.batch_input_directory_ =
            ptree.get<std::string>("files.batch_input_directory", ui_descriptor_.batch_input_directory_);

        image_rendering_action->setChecked(
            !ptree.get<bool>("image_rendering.hidden", image_rendering_group_box->isHidden()));

        const float z_step = ptree.get<float>("image_rendering.z_step", ui_descriptor_.z_step_);
        if (z_step > 0.0f)
            set_z_step(z_step);

        view_action->setChecked(!ptree.get<bool>("view.hidden", view_group_box->isHidden()));

        ui_descriptor_.last_img_type_ = ui_descriptor_.holovibes_.get_cd().img_type == ImgType::Composite
                                            ? "Composite image"
                                            : ui_descriptor_.last_img_type_;

        ui.ViewModeComboBox->setCurrentIndex(static_cast<int>(ui_descriptor_.holovibes_.get_cd().img_type.load()));

        ui_descriptor_.displayAngle = ptree.get("view.mainWindow_rotate", ui_descriptor_.displayAngle);
        ui_descriptor_.xzAngle = ptree.get<float>("view.xCut_rotate", ui_descriptor_.xzAngle);
        ui_descriptor_.yzAngle = ptree.get<float>("view.yCut_rotate", ui_descriptor_.yzAngle);
        ui_descriptor_.displayFlip = ptree.get("view.mainWindow_flip", ui_descriptor_.displayFlip);
        ui_descriptor_.xzFlip = ptree.get("view.xCut_flip", ui_descriptor_.xzFlip);
        ui_descriptor_.yzFlip = ptree.get("view.yCut_flip", ui_descriptor_.yzFlip);

        ui_descriptor_.auto_scale_point_threshold_ =
            ptree.get<size_t>("chart.auto_scale_point_threshold", ui_descriptor_.auto_scale_point_threshold_);

        const uint record_frame_step = ptree.get<uint>("record.record_frame_step", ui_descriptor_.record_frame_step_);
        set_record_frame_step(record_frame_step);

        import_export_action->setChecked(!ptree.get<bool>("import_export.hidden", import_group_box->isHidden()));

        ui.ImportInputFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));

        info_action->setChecked(!ptree.get<bool>("info.hidden", info_group_box->isHidden()));
        ui_descriptor_.theme_index_ = ptree.get<int>("info.theme_type", ui_descriptor_.theme_index_);

        ui_descriptor_.window_max_size = ptree.get<uint>("display.main_window_max_size", 768);
        ui_descriptor_.time_transformation_cuts_window_max_size =
            ptree.get<uint>("display.time_transformation_cuts_window_max_size", 512);
        ui_descriptor_.auxiliary_window_max_size = ptree.get<uint>("display.auxiliary_window_max_size", 512);

        notify();
    }
}

void MainWindow::save_ini(const std::string& path)
{
    LOG_INFO;
    boost::property_tree::ptree ptree;
    GroupBox* image_rendering_group_box = ui.ImageRenderingGroupBox;
    GroupBox* view_group_box = ui.ViewGroupBox;
    Frame* import_export_frame = ui.ImportExportFrame;
    GroupBox* info_group_box = ui.InfoGroupBox;
    Config& config = global::global_config;

    // Save general compute data
    ini::save_ini(ptree, ui_descriptor_.holovibes_.get_cd());

    // Save window specific data
    ptree.put<std::string>("files.default_output_filename", ui_descriptor_.default_output_filename_);
    ptree.put<std::string>("files.record_output_directory", ui_descriptor_.record_output_directory_);
    ptree.put<std::string>("files.file_input_directory", ui_descriptor_.file_input_directory_);
    ptree.put<std::string>("files.batch_input_directory", ui_descriptor_.batch_input_directory_);

    ptree.put<bool>("image_rendering.hidden", image_rendering_group_box->isHidden());

    ptree.put<int>("image_rendering.camera", static_cast<int>(ui_descriptor_.kCamera));

    ptree.put<double>("image_rendering.z_step", ui_descriptor_.z_step_);

    ptree.put<bool>("view.hidden", view_group_box->isHidden());

    ptree.put<float>("view.mainWindow_rotate", ui_descriptor_.displayAngle);
    ptree.put<float>("view.xCut_rotate", ui_descriptor_.xzAngle);
    ptree.put<float>("view.yCut_rotate", ui_descriptor_.yzAngle);
    ptree.put<int>("view.mainWindow_flip", ui_descriptor_.displayFlip);
    ptree.put<int>("view.xCut_flip", ui_descriptor_.xzFlip);
    ptree.put<int>("view.yCut_flip", ui_descriptor_.yzFlip);

    ptree.put<size_t>("chart.auto_scale_point_threshold", ui_descriptor_.auto_scale_point_threshold_);

    ptree.put<uint>("record.record_frame_step", ui_descriptor_.record_frame_step_);

    ptree.put<bool>("import_export.hidden", import_export_frame->isHidden());

    ptree.put<bool>("info.hidden", info_group_box->isHidden());
    ptree.put<ushort>("info.theme_type", ui_descriptor_.theme_index_);

    ptree.put<uint>("display.main_window_max_size", ui_descriptor_.window_max_size);
    ptree.put<uint>("display.time_transformation_cuts_window_max_size",
                    ui_descriptor_.time_transformation_cuts_window_max_size);
    ptree.put<uint>("display.auxiliary_window_max_size", ui_descriptor_.auxiliary_window_max_size);

    boost::property_tree::write_ini(path, ptree);

    LOG_INFO << "Configuration file holovibes.ini overwritten at " << path << std::endl;
}

void MainWindow::open_file(const std::string& path)
{
    LOG_INFO;
    QDesktopServices::openUrl(QUrl::fromLocalFile(QString(path.c_str())));
}
#pragma endregion
/* ------------ */
#pragma region Close Compute

void MainWindow::camera_none()
{
    LOG_INFO;
    ::holovibes::api::close_windows(ui_descriptor_.holovibes_,
                                    ui_descriptor_.mainDisplay,
                                    ui_descriptor_.sliceXZ,
                                    ui_descriptor_.sliceYZ,
                                    ui_descriptor_.lens_window,
                                    ui_descriptor_.raw_window,
                                    ui_descriptor_.filter2d_window,
                                    ui_descriptor_.plot_window_);
    ::holovibes::api::camera_none(ui_descriptor_);

    // Make camera's settings menu unaccessible
    ui.actionSettings->setEnabled(false);

    notify();
}

void MainWindow::reset()
{
    LOG_INFO;
    Config& config = global::global_config;
    int device = 0;

    ::holovibes::api::close_critical_compute(ui_descriptor_);
    camera_none();
    qApp->processEvents();
    if (!::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        ui_descriptor_.holovibes_.stop_compute();
    ui_descriptor_.holovibes_.stop_frame_read();
    ui_descriptor_.holovibes_.get_cd().pindex = 0;
    ui_descriptor_.holovibes_.get_cd().time_transformation_size = 1;
    ui_descriptor_.is_enabled_camera_ = false;
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
    ::holovibes::api::close_windows(ui_descriptor_.holovibes_,
                                    ui_descriptor_.mainDisplay,
                                    ui_descriptor_.sliceXZ,
                                    ui_descriptor_.sliceYZ,
                                    ui_descriptor_.lens_window,
                                    ui_descriptor_.raw_window,
                                    ui_descriptor_.filter2d_window,
                                    ui_descriptor_.plot_window_);
    ::holovibes::api::remove_infos();
    ui_descriptor_.holovibes_.reload_streams();
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
    LOG_INFO;
    ::holovibes::api::close_windows(ui_descriptor_.holovibes_,
                                    ui_descriptor_.mainDisplay,
                                    ui_descriptor_.sliceXZ,
                                    ui_descriptor_.sliceYZ,
                                    ui_descriptor_.lens_window,
                                    ui_descriptor_.raw_window,
                                    ui_descriptor_.filter2d_window,
                                    ui_descriptor_.plot_window_);
    if (!ui_descriptor_.holovibes_.get_cd().is_computation_stopped)
        ::holovibes::api::close_critical_compute(ui_descriptor_);
    camera_none();
    ::holovibes::api::remove_infos();
    save_ini(::holovibes::ini::get_global_ini_path());
}
#pragma endregion
/* ------------ */
#pragma region Cameras
void MainWindow::change_camera(CameraKind c)
{
    LOG_INFO;
    camera_none();

    if (c != CameraKind::NONE)
    {
        try
        {

            ::holovibes::api::change_camera(*this,
                                            ui_descriptor_.holovibes_,
                                            c,
                                            ui_descriptor_.kCamera,
                                            ui_descriptor_.is_enabled_camera_,
                                            ui_descriptor_.import_type_,
                                            ui_descriptor_.mainDisplay,
                                            ui.ImageModeComboBox->currentIndex());

            // Make camera's settings menu accessible
            QAction* settings = ui.actionSettings;
            settings->setEnabled(true);

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

void MainWindow::camera_ids()
{
    LOG_INFO;
    change_camera(CameraKind::IDS);
}

void MainWindow::camera_phantom()
{
    LOG_INFO;
    change_camera(CameraKind::Phantom);
}

void MainWindow::camera_bitflow_cyton()
{
    LOG_INFO;
    change_camera(CameraKind::BitflowCyton);
}

void MainWindow::camera_hamamatsu()
{
    LOG_INFO;
    change_camera(CameraKind::Hamamatsu);
}

void MainWindow::camera_adimec()
{
    LOG_INFO;
    change_camera(CameraKind::Adimec);
}

void MainWindow::camera_xiq()
{
    LOG_INFO;
    change_camera(CameraKind::xiQ);
}

void MainWindow::camera_xib()
{
    LOG_INFO;
    change_camera(CameraKind::xiB);
}

void MainWindow::configure_camera()
{
    LOG_INFO;
    open_file(std::filesystem::current_path().generic_string() + "/" + ui_descriptor_.holovibes_.get_camera_ini_path());
}
#pragma endregion
/* ------------ */
#pragma region Image Mode
void MainWindow::init_image_mode(QPoint& position, QSize& size)
{
    LOG_INFO;
    if (ui_descriptor_.mainDisplay)
    {
        position = ui_descriptor_.mainDisplay->framePosition();
        size = ui_descriptor_.mainDisplay->size();
        ui_descriptor_.mainDisplay.reset(nullptr);
    }
}

void MainWindow::set_raw_mode()
{
    LOG_INFO;
    ::holovibes::api::close_windows(ui_descriptor_.holovibes_,
                                    ui_descriptor_.mainDisplay,
                                    ui_descriptor_.sliceXZ,
                                    ui_descriptor_.sliceYZ,
                                    ui_descriptor_.lens_window,
                                    ui_descriptor_.raw_window,
                                    ui_descriptor_.filter2d_window,
                                    ui_descriptor_.plot_window_);
    ::holovibes::api::close_critical_compute(ui_descriptor_);

    if (ui_descriptor_.is_enabled_camera_)
    {
        QPoint pos(0, 0);
        const FrameDescriptor& fd = ui_descriptor_.holovibes_.get_gpu_input_queue()->get_fd();
        unsigned short width = fd.width;
        unsigned short height = fd.height;
        get_good_size(width, height, ui_descriptor_.window_max_size);
        QSize size(width, height);
        init_image_mode(pos, size);
        ui_descriptor_.holovibes_.get_cd().compute_mode = Computation::Raw;
        createPipe();
        ui_descriptor_.mainDisplay.reset(
            new RawWindow(pos, size, ui_descriptor_.holovibes_.get_gpu_input_queue().get()));
        ui_descriptor_.mainDisplay->setTitle(QString("XY view"));
        ui_descriptor_.mainDisplay->setCd(&ui_descriptor_.holovibes_.get_cd());
        ui_descriptor_.mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));
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
    LOG_INFO;
    try
    {
        ui_descriptor_.holovibes_.start_compute();
        ui_descriptor_.holovibes_.get_compute_pipe()->register_observer(*this);
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "cannot create Pipe: " << e.what();
    }
}

void MainWindow::createHoloWindow()
{
    LOG_INFO;
    QPoint pos(0, 0);
    const FrameDescriptor& fd = ui_descriptor_.holovibes_.get_gpu_input_queue()->get_fd();
    unsigned short width = fd.width;
    unsigned short height = fd.height;
    get_good_size(width, height, ui_descriptor_.window_max_size);
    QSize size(width, height);
    init_image_mode(pos, size);
    /* ---------- */
    try
    {
        ui_descriptor_.mainDisplay.reset(new HoloWindow(pos,
                                                        size,
                                                        ui_descriptor_.holovibes_.get_gpu_output_queue().get(),
                                                        ui_descriptor_.holovibes_.get_compute_pipe(),
                                                        ui_descriptor_.sliceXZ,
                                                        ui_descriptor_.sliceYZ,
                                                        this));
        ui_descriptor_.mainDisplay->set_is_resize(false);
        ui_descriptor_.mainDisplay->setTitle(QString("XY view"));
        ui_descriptor_.mainDisplay->setCd(&ui_descriptor_.holovibes_.get_cd());
        ui_descriptor_.mainDisplay->resetTransform();
        ui_descriptor_.mainDisplay->setAngle(ui_descriptor_.displayAngle);
        ui_descriptor_.mainDisplay->setFlip(ui_descriptor_.displayFlip);
        ui_descriptor_.mainDisplay->setRatio(static_cast<float>(width) / static_cast<float>(height));
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << "createHoloWindow: " << e.what();
    }
}

void MainWindow::set_holographic_mode()
{
    LOG_INFO;
    // That function is used to reallocate the buffers since the Square
    // input mode could have changed
    /* Close windows & destory thread compute */
    ::holovibes::api::close_windows(ui_descriptor_.holovibes_,
                                    ui_descriptor_.mainDisplay,
                                    ui_descriptor_.sliceXZ,
                                    ui_descriptor_.sliceYZ,
                                    ui_descriptor_.lens_window,
                                    ui_descriptor_.raw_window,
                                    ui_descriptor_.filter2d_window,
                                    ui_descriptor_.plot_window_);
    ::holovibes::api::close_critical_compute(ui_descriptor_);

    /* ---------- */
    try
    {
        ui_descriptor_.holovibes_.get_cd().compute_mode = Computation::Hologram;
        /* Pipe & Window */
        createPipe();
        createHoloWindow();
        /* Info Manager */
        const FrameDescriptor& fd = ui_descriptor_.holovibes_.get_gpu_output_queue()->get_fd();
        std::string fd_info =
            std::to_string(fd.width) + "x" + std::to_string(fd.height) + " - " + std::to_string(fd.depth * 8) + "bit";
        Holovibes::instance().get_info_container().add_indication(InformationContainer::IndicationType::OUTPUT_FORMAT,
                                                                  fd_info);
        /* Contrast */
        ui_descriptor_.holovibes_.get_cd().contrast_enabled = true;

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

void MainWindow::refreshViewMode()
{
    LOG_INFO;
    float old_scale = 1.f;
    glm::vec2 old_translation(0.f, 0.f);
    if (ui_descriptor_.mainDisplay)
    {
        old_scale = ui_descriptor_.mainDisplay->getScale();
        old_translation = ui_descriptor_.mainDisplay->getTranslate();
    }
    ::holovibes::api::close_windows(ui_descriptor_.holovibes_,
                                    ui_descriptor_.mainDisplay,
                                    ui_descriptor_.sliceXZ,
                                    ui_descriptor_.sliceYZ,
                                    ui_descriptor_.lens_window,
                                    ui_descriptor_.raw_window,
                                    ui_descriptor_.filter2d_window,
                                    ui_descriptor_.plot_window_);
    ::holovibes::api::close_critical_compute(ui_descriptor_);
    ui_descriptor_.holovibes_.get_cd().img_type = static_cast<ImgType>(ui.ViewModeComboBox->currentIndex());
    try
    {
        createPipe();
        createHoloWindow();
        ui_descriptor_.mainDisplay->setScale(old_scale);
        ui_descriptor_.mainDisplay->setTranslate(old_translation[0], old_translation[1]);
    }
    catch (const std::runtime_error& e)
    {
        ui_descriptor_.mainDisplay.reset(nullptr);
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
void MainWindow::set_view_mode(const QString value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (need_refresh(ui_descriptor_.last_img_type_, value))
    {
        refreshViewMode();
        if (ui_descriptor_.holovibes_.get_cd().img_type == ImgType::Composite)
        {
            const unsigned min_val_composite = ui_descriptor_.holovibes_.get_cd().time_transformation_size == 1 ? 0 : 1;
            const unsigned max_val_composite = ui_descriptor_.holovibes_.get_cd().time_transformation_size - 1;

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
    ui_descriptor_.last_img_type_ = value;

    auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get());

    pipe->insert_fn_end_vect([=]() {
        ui_descriptor_.holovibes_.get_cd().img_type = static_cast<ImgType>(ui.ViewModeComboBox->currentIndex());
        notify();
        layout_toggled();
    });
    pipe_refresh();

    // Force XYview autocontrast
    pipe->autocontrast_end_pipe(WindowKind::XYview);
    // Force cuts views autocontrast if needed
    if (ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled)
        set_auto_contrast_cuts();
}

void MainWindow::set_image_mode(QString mode)
{
    LOG_INFO;
    const bool is_null_mode = (mode == nullptr);
    ::holovibes::api::set_image_mode(*this,
                                     ui_descriptor_.holovibes_,
                                     is_null_mode,
                                     ui.ImageModeComboBox->currentIndex());
}
#pragma endregion

#pragma region Batch

static void adapt_time_transformation_stride_to_batch_size(ComputeDescriptor& cd)
{
    if (cd.time_transformation_stride < cd.batch_size)
        cd.time_transformation_stride = cd.batch_size.load();
    // Go to lower multiple
    if (cd.time_transformation_stride % cd.batch_size != 0)
        cd.time_transformation_stride -= cd.time_transformation_stride % cd.batch_size;
}

void MainWindow::update_batch_size()
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    int value = ui.BatchSizeSpinBox->value();

    if (value == ui_descriptor_.holovibes_.get_cd().batch_size)
        return;

    auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            ui_descriptor_.holovibes_.get_cd().batch_size = value;
            adapt_time_transformation_stride_to_batch_size(ui_descriptor_.holovibes_.get_cd());
            ui_descriptor_.holovibes_.get_compute_pipe()->request_update_batch_size();
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
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    int value = ui.TimeTransformationStrideSpinBox->value();

    if (value == ui_descriptor_.holovibes_.get_cd().time_transformation_stride)
        return;

    auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            ui_descriptor_.holovibes_.get_cd().time_transformation_stride = value;
            adapt_time_transformation_stride_to_batch_size(ui_descriptor_.holovibes_.get_cd());
            ui_descriptor_.holovibes_.get_compute_pipe()->request_update_time_transformation_stride();
            ui.NumberOfFramesSpinBox->setValue(
                ceil((ui.ImportEndIndexSpinBox->value() - ui.ImportStartIndexSpinBox->value()) /
                     (float)ui.TimeTransformationStrideSpinBox->value()));
            notify();
        });
    }
    else
        LOG_INFO << "COULD NOT GET PIPE" << std::endl;
}

void MainWindow::toggle_time_transformation_cuts(bool checked)
{
    LOG_INFO;
    QComboBox* winSelection = ui.WindowSelectionComboBox;
    winSelection->setEnabled(checked);
    winSelection->setCurrentIndex((!checked) ? 0 : winSelection->currentIndex());
    if (checked)
    {
        try
        {
            ui_descriptor_.holovibes_.get_compute_pipe()->create_stft_slice_queue();
            // set positions of new windows according to the position of the
            // main GL window
            QPoint xzPos =
                ui_descriptor_.mainDisplay->framePosition() + QPoint(0, ui_descriptor_.mainDisplay->height() + 42);
            QPoint yzPos =
                ui_descriptor_.mainDisplay->framePosition() + QPoint(ui_descriptor_.mainDisplay->width() + 20, 0);
            const ushort nImg = ui_descriptor_.holovibes_.get_cd().time_transformation_size;
            uint time_transformation_size = std::max(256u, std::min(512u, (uint)nImg));

            if (time_transformation_size > ui_descriptor_.time_transformation_cuts_window_max_size)
                time_transformation_size = ui_descriptor_.time_transformation_cuts_window_max_size;

            while (ui_descriptor_.holovibes_.get_compute_pipe()->get_update_time_transformation_size_request())
                continue;
            while (ui_descriptor_.holovibes_.get_compute_pipe()->get_cuts_request())
                continue;
            ui_descriptor_.sliceXZ.reset(
                new SliceWindow(xzPos,
                                QSize(ui_descriptor_.mainDisplay->width(), time_transformation_size),
                                ui_descriptor_.holovibes_.get_compute_pipe()->get_stft_slice_queue(0).get(),
                                KindOfView::SliceXZ,
                                this));
            ui_descriptor_.sliceXZ->setTitle("XZ view");
            ui_descriptor_.sliceXZ->setAngle(ui_descriptor_.xzAngle);
            ui_descriptor_.sliceXZ->setFlip(ui_descriptor_.xzFlip);
            ui_descriptor_.sliceXZ->setCd(&ui_descriptor_.holovibes_.get_cd());

            ui_descriptor_.sliceYZ.reset(
                new SliceWindow(yzPos,
                                QSize(time_transformation_size, ui_descriptor_.mainDisplay->height()),
                                ui_descriptor_.holovibes_.get_compute_pipe()->get_stft_slice_queue(1).get(),
                                KindOfView::SliceYZ,
                                this));
            ui_descriptor_.sliceYZ->setTitle("YZ view");
            ui_descriptor_.sliceYZ->setAngle(ui_descriptor_.yzAngle);
            ui_descriptor_.sliceYZ->setFlip(ui_descriptor_.yzFlip);
            ui_descriptor_.sliceYZ->setCd(&ui_descriptor_.holovibes_.get_cd());

            ui_descriptor_.mainDisplay->getOverlayManager().create_overlay<Cross>();
            ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled = true;
            set_auto_contrast_cuts();
            auto holo = dynamic_cast<HoloWindow*>(ui_descriptor_.mainDisplay.get());
            if (holo)
                holo->update_slice_transforms();
            notify();
        }
        catch (const std::logic_error& e)
        {
            LOG_ERROR << e.what() << std::endl;
            cancel_time_transformation_cuts();
        }
    }
    else
    {
        cancel_time_transformation_cuts();
    }
}

void MainWindow::cancel_time_transformation_cuts()
{
    LOG_INFO;
    if (ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled)
    {
        std::function<void()> callback = []() { return; };

        if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get()))
        {
            callback = ([=]() {
                ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled = false;
                pipe->delete_stft_slice_queue();

                ui.TimeTransformationCutsCheckBox->setChecked(false);
                notify();
            });
        }
        ::holovibes::api::cancel_time_transformation_cuts(ui_descriptor_, callback);
        ui_descriptor_.sliceXZ.reset(nullptr);
        ui_descriptor_.sliceYZ.reset(nullptr);

        if (ui_descriptor_.mainDisplay)
        {
            ui_descriptor_.mainDisplay->setCursor(Qt::ArrowCursor);
            ui_descriptor_.mainDisplay->getOverlayManager().disable_all(SliceCross);
            ui_descriptor_.mainDisplay->getOverlayManager().disable_all(Cross);
        }
    }
    notify();
}

#pragma endregion
/* ------------ */
#pragma region Computation
void MainWindow::change_window()
{
    LOG_INFO;
    QComboBox* window_cbox = ui.WindowSelectionComboBox;

    if (window_cbox->currentIndex() == 0)
        ui_descriptor_.holovibes_.get_cd().current_window = WindowKind::XYview;
    else if (window_cbox->currentIndex() == 1)
        ui_descriptor_.holovibes_.get_cd().current_window = WindowKind::XZview;
    else if (window_cbox->currentIndex() == 2)
        ui_descriptor_.holovibes_.get_cd().current_window = WindowKind::YZview;
    else if (window_cbox->currentIndex() == 3)
        ui_descriptor_.holovibes_.get_cd().current_window = WindowKind::Filter2D;
    pipe_refresh();
    notify();
}

void MainWindow::toggle_renormalize(bool value)
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().renorm_enabled = value;

    ui_descriptor_.holovibes_.get_compute_pipe()->request_clear_img_acc();
    pipe_refresh();
}

void MainWindow::set_filter2d(bool checked)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (!checked)
    {
        ui_descriptor_.holovibes_.get_cd().filter2d_enabled = checked;
        cancel_filter2d();
    }
    else
    {
        const camera::FrameDescriptor& fd = ui_descriptor_.holovibes_.get_gpu_input_queue()->get_fd();

        // Set the input box related to the filter2d
        ui.Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));
        set_filter2d_n2(ui.Filter2DN2SpinBox->value());
        set_filter2d_n1(ui.Filter2DN1SpinBox->value());

        if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get()))
            pipe->autocontrast_end_pipe(WindowKind::XYview);
        ui_descriptor_.holovibes_.get_cd().filter2d_enabled = checked;
    }
    pipe_refresh();
    notify();
}

void MainWindow::disable_filter2d_view()
{
    LOG_INFO;

    auto pipe = ui_descriptor_.holovibes_.get_compute_pipe();
    pipe->request_disable_filter2d_view();

    // Wait for the filter2d view to be disabled for notify
    while (pipe->get_disable_filter2d_view_requested())
        continue;

    if (ui_descriptor_.filter2d_window)
    {
        // Remove the on triggered event

        disconnect(ui_descriptor_.filter2d_window.get(), SIGNAL(destroyed()), this, SLOT(disable_filter2d_view()));
    }

    // Change the focused window
    change_window();

    notify();
}

void MainWindow::update_filter2d_view(bool checked)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (checked)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos =
                ui_descriptor_.mainDisplay->framePosition() + QPoint(ui_descriptor_.mainDisplay->width() + 310, 0);
            auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get());
            if (pipe)
            {
                pipe->request_filter2d_view();

                const FrameDescriptor& fd = ui_descriptor_.holovibes_.get_gpu_output_queue()->get_fd();
                ushort filter2d_window_width = fd.width;
                ushort filter2d_window_height = fd.height;
                get_good_size(filter2d_window_width, filter2d_window_height, ui_descriptor_.auxiliary_window_max_size);

                // Wait for the filter2d view to be enabled for notify
                while (pipe->get_filter2d_view_requested())
                    continue;

                ui_descriptor_.filter2d_window.reset(
                    new Filter2DWindow(pos,
                                       QSize(filter2d_window_width, filter2d_window_height),
                                       pipe->get_filter2d_view_queue().get(),
                                       this));

                ui_descriptor_.filter2d_window->setTitle("Filter2D view");
                ui_descriptor_.filter2d_window->setCd(&ui_descriptor_.holovibes_.get_cd());

                connect(ui_descriptor_.filter2d_window.get(), SIGNAL(destroyed()), this, SLOT(disable_filter2d_view()));
                ui_descriptor_.holovibes_.get_cd().set_log_scale_slice_enabled(WindowKind::Filter2D, true);
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
        ui_descriptor_.filter2d_window.reset(nullptr);
    }

    pipe_refresh();
    notify();
}

void MainWindow::set_filter2d_n1(int n)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    ui_descriptor_.holovibes_.get_cd().filter2d_n1 = n;

    if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XYview);
        if (ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled)
        {
            pipe->autocontrast_end_pipe(WindowKind::XZview);
            pipe->autocontrast_end_pipe(WindowKind::YZview);
        }
        if (ui_descriptor_.holovibes_.get_cd().filter2d_view_enabled)
            pipe->autocontrast_end_pipe(WindowKind::Filter2D);
    }

    pipe_refresh();
    notify();
}

void MainWindow::set_filter2d_n2(int n)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    ui_descriptor_.holovibes_.get_cd().filter2d_n2 = n;

    if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XYview);
        if (ui_descriptor_.holovibes_.get_cd().time_transformation_cuts_enabled)
        {
            pipe->autocontrast_end_pipe(WindowKind::XZview);
            pipe->autocontrast_end_pipe(WindowKind::YZview);
        }
        if (ui_descriptor_.holovibes_.get_cd().filter2d_view_enabled)
            pipe->autocontrast_end_pipe(WindowKind::Filter2D);
    }

    pipe_refresh();
    notify();
}

void MainWindow::cancel_filter2d()
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (ui_descriptor_.holovibes_.get_cd().filter2d_view_enabled)
        update_filter2d_view(false);
    pipe_refresh();
    notify();
}

void MainWindow::set_fft_shift(const bool value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    ui_descriptor_.holovibes_.get_cd().fft_shift_enabled = value;
    pipe_refresh();
}

void MainWindow::set_time_transformation_size()
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    int time_transformation_size = ui.timeTransformationSizeSpinBox->value();
    time_transformation_size = std::max(1, time_transformation_size);

    if (time_transformation_size == ui_descriptor_.holovibes_.get_cd().time_transformation_size)
        return;
    notify();
    auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get());
    if (pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            ui_descriptor_.holovibes_.get_cd().time_transformation_size = time_transformation_size;
            ui_descriptor_.holovibes_.get_compute_pipe()->request_update_time_transformation_size();
            set_p_accu();
            // This will not do anything until
            // SliceWindow::changeTexture() isn't coded.
        });
    }
}

void MainWindow::update_lens_view(bool value)
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().gpu_lens_display_enabled = value;

    if (value)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos =
                ui_descriptor_.mainDisplay->framePosition() + QPoint(ui_descriptor_.mainDisplay->width() + 310, 0);
            ICompute* pipe = ui_descriptor_.holovibes_.get_compute_pipe().get();

            const FrameDescriptor& fd = ui_descriptor_.holovibes_.get_gpu_input_queue()->get_fd();
            ushort lens_window_width = fd.width;
            ushort lens_window_height = fd.height;
            get_good_size(lens_window_width, lens_window_height, ui_descriptor_.auxiliary_window_max_size);

            ui_descriptor_.lens_window.reset(new RawWindow(pos,
                                                           QSize(lens_window_width, lens_window_height),
                                                           pipe->get_lens_queue().get(),
                                                           KindOfView::Lens));

            ui_descriptor_.lens_window->setTitle("Lens view");
            ui_descriptor_.lens_window->setCd(&ui_descriptor_.holovibes_.get_cd());

            // when the window is destoryed, disable_lens_view() will be triggered
            connect(ui_descriptor_.lens_window.get(), SIGNAL(destroyed()), this, SLOT(disable_lens_view()));
        }
        catch (const std::exception& e)
        {
            LOG_ERROR << e.what() << std::endl;
        }
    }

    else
    {
        disable_lens_view();
        ui_descriptor_.lens_window.reset(nullptr);
    }

    pipe_refresh();
}

void MainWindow::disable_lens_view()
{
    LOG_INFO;
    if (ui_descriptor_.lens_window)
        disconnect(ui_descriptor_.lens_window.get(), SIGNAL(destroyed()), this, SLOT(disable_lens_view()));

    ui_descriptor_.holovibes_.get_cd().gpu_lens_display_enabled = false;
    ui_descriptor_.holovibes_.get_compute_pipe()->request_disable_lens_view();
    notify();
}

void MainWindow::update_raw_view(bool value)
{
    LOG_INFO;
    if (value)
    {
        if (ui_descriptor_.holovibes_.get_cd().batch_size > global::global_config.output_queue_max_size)
        {
            ui.RawDisplayingCheckBox->setChecked(false);
            LOG_ERROR << "[RAW VIEW] Batch size must be lower than output queue size";
            return;
        }

        auto pipe = ui_descriptor_.holovibes_.get_compute_pipe();
        pipe->request_raw_view();

        // Wait for the raw view to be enabled for notify
        while (pipe->get_raw_view_requested())
            continue;

        const FrameDescriptor& fd = ui_descriptor_.holovibes_.get_gpu_input_queue()->get_fd();
        ushort raw_window_width = fd.width;
        ushort raw_window_height = fd.height;
        get_good_size(raw_window_width, raw_window_height, ui_descriptor_.auxiliary_window_max_size);

        // set positions of new windows according to the position of the main GL
        // window and Lens window
        QPoint pos = ui_descriptor_.mainDisplay->framePosition() + QPoint(ui_descriptor_.mainDisplay->width() + 310, 0);
        ui_descriptor_.raw_window.reset(
            new RawWindow(pos, QSize(raw_window_width, raw_window_height), pipe->get_raw_view_queue().get()));

        ui_descriptor_.raw_window->setTitle("Raw view");
        ui_descriptor_.raw_window->setCd(&ui_descriptor_.holovibes_.get_cd());

        connect(ui_descriptor_.raw_window.get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));
    }
    else
    {
        ui_descriptor_.raw_window.reset(nullptr);
        disable_raw_view();
    }
    pipe_refresh();
}

void MainWindow::disable_raw_view()
{
    LOG_INFO;
    if (ui_descriptor_.raw_window)
        disconnect(ui_descriptor_.raw_window.get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));

    auto pipe = ui_descriptor_.holovibes_.get_compute_pipe();
    pipe->request_disable_raw_view();

    // Wait for the raw view to be disabled for notify
    while (pipe->get_disable_raw_view_requested())
        continue;

    notify();
}

void MainWindow::set_p_accu()
{
    LOG_INFO;
    auto spinbox = ui.PAccSpinBox;
    auto checkBox = ui.PAccuCheckBox;
    ui_descriptor_.holovibes_.get_cd().p_accu_enabled = checkBox->isChecked();
    ui_descriptor_.holovibes_.get_cd().p_acc_level = spinbox->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_x_accu()
{
    LOG_INFO;
    auto box = ui.XAccSpinBox;
    auto checkBox = ui.XAccuCheckBox;
    ui_descriptor_.holovibes_.get_cd().x_accu_enabled = checkBox->isChecked();
    ui_descriptor_.holovibes_.get_cd().x_acc_level = box->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_y_accu()
{
    LOG_INFO;
    auto box = ui.YAccSpinBox;
    auto checkBox = ui.YAccuCheckBox;
    ui_descriptor_.holovibes_.get_cd().y_accu_enabled = checkBox->isChecked();
    ui_descriptor_.holovibes_.get_cd().y_acc_level = box->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_x_y()
{
    LOG_INFO;
    auto& fd = ui_descriptor_.holovibes_.get_gpu_input_queue()->get_fd();
    uint x = ui.XSpinBox->value();
    uint y = ui.YSpinBox->value();

    if (x < fd.width)
        ui_descriptor_.holovibes_.get_cd().x_cuts = x;

    if (y < fd.height)
        ui_descriptor_.holovibes_.get_cd().y_cuts = y;
}

void MainWindow::set_q(int value)
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().q_index = value;
    notify();
}

void MainWindow::set_q_acc()
{
    LOG_INFO;
    auto spinbox = ui.Q_AccSpinBox;
    auto checkBox = ui.Q_AccuCheckBox;
    ui_descriptor_.holovibes_.get_cd().q_acc_enabled = checkBox->isChecked();
    ui_descriptor_.holovibes_.get_cd().q_acc_level = spinbox->value();
    notify();
}

void MainWindow::set_p(int value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (value < static_cast<int>(ui_descriptor_.holovibes_.get_cd().time_transformation_size))
    {
        ui_descriptor_.holovibes_.get_cd().pindex = value;
        pipe_refresh();
        notify();
    }
    else
        LOG_ERROR << "p param has to be between 1 and #img";
}

void MainWindow::set_composite_intervals()
{
    LOG_INFO;
    // PRedSpinBox_Composite value cannont be higher than PBlueSpinBox_Composite
    ui.PRedSpinBox_Composite->setValue(std::min(ui.PRedSpinBox_Composite->value(), ui.PBlueSpinBox_Composite->value()));
    ui_descriptor_.holovibes_.get_cd().composite_p_red = ui.PRedSpinBox_Composite->value();
    ui_descriptor_.holovibes_.get_cd().composite_p_blue = ui.PBlueSpinBox_Composite->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_h_min()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_p_min_h = ui.SpinBox_hue_freq_min->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_h_max()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_p_max_h = ui.SpinBox_hue_freq_max->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_s_min()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_p_min_s = ui.SpinBox_saturation_freq_min->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_s_max()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_p_max_s = ui.SpinBox_saturation_freq_max->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_v_min()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_p_min_v = ui.SpinBox_value_freq_min->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_v_max()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_p_max_v = ui.SpinBox_value_freq_max->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_weights()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().weight_r = ui.WeightSpinBox_R->value();
    ui_descriptor_.holovibes_.get_cd().weight_g = ui.WeightSpinBox_G->value();
    ui_descriptor_.holovibes_.get_cd().weight_b = ui.WeightSpinBox_B->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_auto_weights(bool value)
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_auto_weights_ = value;
    set_auto_contrast();
}

void MainWindow::click_composite_rgb_or_hsv()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_kind =
        ui.radioButton_rgb->isChecked() ? CompositeKind::RGB : CompositeKind::HSV;
    if (ui.radioButton_rgb->isChecked())
    {
        ui.PRedSpinBox_Composite->setValue(ui.SpinBox_hue_freq_min->value());
        ui.PBlueSpinBox_Composite->setValue(ui.SpinBox_hue_freq_max->value());
    }
    else
    {
        ui.SpinBox_hue_freq_min->setValue(ui.PRedSpinBox_Composite->value());
        ui.SpinBox_hue_freq_max->setValue(ui.PBlueSpinBox_Composite->value());
        ui.SpinBox_saturation_freq_min->setValue(ui.PRedSpinBox_Composite->value());
        ui.SpinBox_saturation_freq_max->setValue(ui.PBlueSpinBox_Composite->value());
        ui.SpinBox_value_freq_min->setValue(ui.PRedSpinBox_Composite->value());
        ui.SpinBox_value_freq_max->setValue(ui.PBlueSpinBox_Composite->value());
    }
    notify();
}

void MainWindow::actualize_frequency_channel_s()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_p_activated_s = ui.checkBox_saturation_freq->isChecked();
    ui.SpinBox_saturation_freq_min->setDisabled(!ui.checkBox_saturation_freq->isChecked());
    ui.SpinBox_saturation_freq_max->setDisabled(!ui.checkBox_saturation_freq->isChecked());
}

void MainWindow::actualize_frequency_channel_v()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().composite_p_activated_v = ui.checkBox_value_freq->isChecked();
    ui.SpinBox_value_freq_min->setDisabled(!ui.checkBox_value_freq->isChecked());
    ui.SpinBox_value_freq_max->setDisabled(!ui.checkBox_value_freq->isChecked());
}

void MainWindow::actualize_checkbox_h_gaussian_blur()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().h_blur_activated = ui.checkBox_h_gaussian_blur->isChecked();
    ui.SpinBox_hue_blur_kernel_size->setEnabled(ui.checkBox_h_gaussian_blur->isChecked());
}

void MainWindow::actualize_kernel_size_blur()
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().h_blur_kernel_size = ui.SpinBox_hue_blur_kernel_size->value();
}

void fancy_Qslide_text_percent(char* str)
{
    size_t len = strlen(str);
    if (len < 2)
    {
        str[1] = str[0];
        str[0] = '0';
        str[2] = '\0';
        len = 2;
    }
    str[len] = str[len - 1];
    str[len - 1] = '.';
    str[len + 1] = '%';
    str[len + 2] = '\0';
}

void slide_update_threshold(QSlider& slider,
                            std::atomic<float>& receiver,
                            std::atomic<float>& bound_to_update,
                            QSlider& slider_to_update,
                            QLabel& to_be_written_in,
                            std::atomic<float>& lower_bound,
                            std::atomic<float>& upper_bound)
{
    // Store the slider value in ui_descriptor_.holovibes_.get_cd() (ComputeDescriptor)
    receiver = slider.value() / 1000.0f;

    char array[10];
    sprintf_s(array, "%d", slider.value());
    fancy_Qslide_text_percent(array);
    to_be_written_in.setText(QString(array));

    if (lower_bound > upper_bound)
    {
        // FIXME bound_to_update = receiver ?
        bound_to_update = slider.value() / 1000.0f;

        slider_to_update.setValue(slider.value());
    }
}

void MainWindow::slide_update_threshold_h_min()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_hue_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_h_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_h_threshold_max,
                           *ui.horizontalSlider_hue_threshold_max,
                           *ui.label_hue_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_h_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_h_threshold_max);
}

void MainWindow::slide_update_threshold_h_max()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_hue_threshold_max,
                           ui_descriptor_.holovibes_.get_cd().slider_h_threshold_max,
                           ui_descriptor_.holovibes_.get_cd().slider_h_threshold_min,
                           *ui.horizontalSlider_hue_threshold_min,
                           *ui.label_hue_threshold_max,
                           ui_descriptor_.holovibes_.get_cd().slider_h_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_h_threshold_max);
}

void MainWindow::slide_update_threshold_s_min()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_saturation_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_s_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_s_threshold_max,
                           *ui.horizontalSlider_saturation_threshold_max,
                           *ui.label_saturation_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_s_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_s_threshold_max);
}

void MainWindow::slide_update_threshold_s_max()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_saturation_threshold_max,
                           ui_descriptor_.holovibes_.get_cd().slider_s_threshold_max,
                           ui_descriptor_.holovibes_.get_cd().slider_s_threshold_min,
                           *ui.horizontalSlider_saturation_threshold_min,
                           *ui.label_saturation_threshold_max,
                           ui_descriptor_.holovibes_.get_cd().slider_s_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_s_threshold_max);
}

void MainWindow::slide_update_threshold_v_min()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_value_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_v_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_v_threshold_max,
                           *ui.horizontalSlider_value_threshold_max,
                           *ui.label_value_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_v_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_v_threshold_max);
}

void MainWindow::slide_update_threshold_v_max()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_value_threshold_max,
                           ui_descriptor_.holovibes_.get_cd().slider_v_threshold_max,
                           ui_descriptor_.holovibes_.get_cd().slider_v_threshold_min,
                           *ui.horizontalSlider_value_threshold_min,
                           *ui.label_value_threshold_max,
                           ui_descriptor_.holovibes_.get_cd().slider_v_threshold_min,
                           ui_descriptor_.holovibes_.get_cd().slider_v_threshold_max);
}

void MainWindow::increment_p()
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (ui_descriptor_.holovibes_.get_cd().pindex < ui_descriptor_.holovibes_.get_cd().time_transformation_size)
    {
        ui_descriptor_.holovibes_.get_cd().pindex = ui_descriptor_.holovibes_.get_cd().pindex + 1;
        set_auto_contrast();
        notify();
    }
    else
        LOG_ERROR << "p param has to be between 1 and #img";
}

void MainWindow::decrement_p()
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (ui_descriptor_.holovibes_.get_cd().pindex > 0)
    {
        ui_descriptor_.holovibes_.get_cd().pindex = ui_descriptor_.holovibes_.get_cd().pindex - 1;
        set_auto_contrast();
        notify();
    }
    else
        LOG_ERROR << "p param has to be between 1 and #img";
}

void MainWindow::set_wavelength(const double value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    ui_descriptor_.holovibes_.get_cd().lambda = static_cast<float>(value) * 1.0e-9f;
    pipe_refresh();
}

void MainWindow::set_z(const double value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    ui_descriptor_.holovibes_.get_cd().zdistance = static_cast<float>(value);
    pipe_refresh();
}

void MainWindow::increment_z()
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    set_z(ui_descriptor_.holovibes_.get_cd().zdistance + ui_descriptor_.z_step_);
    ui.ZDoubleSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().zdistance);
}

void MainWindow::decrement_z()
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    set_z(ui_descriptor_.holovibes_.get_cd().zdistance - ui_descriptor_.z_step_);
    ui.ZDoubleSpinBox->setValue(ui_descriptor_.holovibes_.get_cd().zdistance);
}

void MainWindow::set_z_step(const double value)
{
    LOG_INFO;
    ui_descriptor_.z_step_ = value;
    ui.ZDoubleSpinBox->setSingleStep(value);
}

void MainWindow::set_space_transformation(const QString value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (value == "None")
        ui_descriptor_.holovibes_.get_cd().space_transformation = SpaceTransformation::None;
    else if (value == "1FFT")
        ui_descriptor_.holovibes_.get_cd().space_transformation = SpaceTransformation::FFT1;
    else if (value == "2FFT")
        ui_descriptor_.holovibes_.get_cd().space_transformation = SpaceTransformation::FFT2;
    else
    {
        // Shouldn't happen
        ui_descriptor_.holovibes_.get_cd().space_transformation = SpaceTransformation::None;
        LOG_ERROR << "Unknown space transform: " << value.toStdString() << ", falling back to None";
    }
    set_holographic_mode();
}

void MainWindow::set_time_transformation(QString value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (value == "STFT")
        ui_descriptor_.holovibes_.get_cd().time_transformation = TimeTransformation::STFT;
    else if (value == "PCA")
        ui_descriptor_.holovibes_.get_cd().time_transformation = TimeTransformation::PCA;
    else if (value == "None")
        ui_descriptor_.holovibes_.get_cd().time_transformation = TimeTransformation::NONE;
    else if (value == "SSA_STFT")
        ui_descriptor_.holovibes_.get_cd().time_transformation = TimeTransformation::SSA_STFT;
    set_holographic_mode();
}

void MainWindow::set_unwrapping_2d(const bool value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    ui_descriptor_.holovibes_.get_compute_pipe()->request_unwrapping_2d(value);
    pipe_refresh();
    notify();
}

void MainWindow::set_accumulation(bool value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    ui_descriptor_.holovibes_.get_cd().set_accumulation(ui_descriptor_.holovibes_.get_cd().current_window, value);
    pipe_refresh();
    notify();
}

void MainWindow::set_accumulation_level(int value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    ui_descriptor_.holovibes_.get_cd().set_accumulation_level(ui_descriptor_.holovibes_.get_cd().current_window, value);
    pipe_refresh();
}

void MainWindow::pipe_refresh()
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    try
    {
        // FIXME: Should better not use a if structure with 2 method access, 1 dereferencing, and 1 negation bitwise
        // operation to set a boolean
        // But maybe a simple read access that create a false condition result is better than simply making a
        // writting access
        if (!ui_descriptor_.holovibes_.get_compute_pipe()->get_request_refresh())
            ui_descriptor_.holovibes_.get_compute_pipe()->request_refresh();
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what();
    }
}

void MainWindow::set_composite_area()
{
    LOG_INFO;
    ui_descriptor_.mainDisplay->getOverlayManager().create_overlay<CompositeArea>();
}

#pragma endregion
/* ------------ */
#pragma region Texture
void MainWindow::rotateTexture()
{
    LOG_INFO;
    WindowKind curWin = ui_descriptor_.holovibes_.get_cd().current_window;

    if (curWin == WindowKind::XYview)
    {
        ui_descriptor_.displayAngle = (ui_descriptor_.displayAngle == 270.f) ? 0.f : ui_descriptor_.displayAngle + 90.f;
        ui_descriptor_.mainDisplay->setAngle(ui_descriptor_.displayAngle);
    }
    else if (ui_descriptor_.sliceXZ && curWin == WindowKind::XZview)
    {
        ui_descriptor_.xzAngle = (ui_descriptor_.xzAngle == 270.f) ? 0.f : ui_descriptor_.xzAngle + 90.f;
        ui_descriptor_.sliceXZ->setAngle(ui_descriptor_.xzAngle);
    }
    else if (ui_descriptor_.sliceYZ && curWin == WindowKind::YZview)
    {
        ui_descriptor_.yzAngle = (ui_descriptor_.yzAngle == 270.f) ? 0.f : ui_descriptor_.yzAngle + 90.f;
        ui_descriptor_.sliceYZ->setAngle(ui_descriptor_.yzAngle);
    }
    notify();
}

void MainWindow::flipTexture()
{
    LOG_INFO;
    WindowKind curWin = ui_descriptor_.holovibes_.get_cd().current_window;

    if (curWin == WindowKind::XYview)
    {
        ui_descriptor_.displayFlip = !ui_descriptor_.displayFlip;
        ui_descriptor_.mainDisplay->setFlip(ui_descriptor_.displayFlip);
    }
    else if (ui_descriptor_.sliceXZ && curWin == WindowKind::XZview)
    {
        ui_descriptor_.xzFlip = !ui_descriptor_.xzFlip;
        ui_descriptor_.sliceXZ->setFlip(ui_descriptor_.xzFlip);
    }
    else if (ui_descriptor_.sliceYZ && curWin == WindowKind::YZview)
    {
        ui_descriptor_.yzFlip = !ui_descriptor_.yzFlip;
        ui_descriptor_.sliceYZ->setFlip(ui_descriptor_.yzFlip);
    }
    notify();
}

#pragma endregion
/* ------------ */
#pragma region Contrast - Log
void MainWindow::set_contrast_mode(bool value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    change_window();
    ui_descriptor_.holovibes_.get_cd().contrast_enabled = value;
    ui_descriptor_.holovibes_.get_cd().contrast_auto_refresh = true;
    pipe_refresh();
    notify();
}

void MainWindow::set_auto_contrast_cuts()
{
    LOG_INFO;
    if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XZview);
        pipe->autocontrast_end_pipe(WindowKind::YZview);
    }
}

void MainWindow::QSpinBoxQuietSetValue(QSpinBox* spinBox, int value)
{
    LOG_INFO;
    spinBox->blockSignals(true);
    spinBox->setValue(value);
    spinBox->blockSignals(false);
}

void MainWindow::QSliderQuietSetValue(QSlider* slider, int value)
{
    LOG_INFO;
    slider->blockSignals(true);
    slider->setValue(value);
    slider->blockSignals(false);
}

void MainWindow::QDoubleSpinBoxQuietSetValue(QDoubleSpinBox* spinBox, double value)
{
    LOG_INFO;
    spinBox->blockSignals(true);
    spinBox->setValue(value);
    spinBox->blockSignals(false);
}

void MainWindow::set_auto_contrast()
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    try
    {
        if (auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get()))
            pipe->autocontrast_end_pipe(ui_descriptor_.holovibes_.get_cd().current_window);
    }
    catch (const std::runtime_error& e)
    {
        LOG_ERROR << e.what() << std::endl;
    }
}

void MainWindow::set_contrast_min(const double value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (ui_descriptor_.holovibes_.get_cd().contrast_enabled)
    {
        // FIXME: type issue, manipulatiion of double casted to float implies lost of data
        // Get the minimum contrast value rounded for the comparison
        const float old_val = ui_descriptor_.holovibes_.get_cd().get_truncate_contrast_min(
            ui_descriptor_.holovibes_.get_cd().current_window);
        // Floating number issue: cast to float for the comparison
        const float val = value;
        if (old_val != val)
        {
            ui_descriptor_.holovibes_.get_cd().set_contrast_min(ui_descriptor_.holovibes_.get_cd().current_window,
                                                                value);
            pipe_refresh();
        }
    }
}

void MainWindow::set_contrast_max(const double value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (ui_descriptor_.holovibes_.get_cd().contrast_enabled)
    {
        // FIXME: type issue, manipulatiion of double casted to float implies lost of data
        // Get the maximum contrast value rounded for the comparison
        const float old_val = ui_descriptor_.holovibes_.get_cd().get_truncate_contrast_max(
            ui_descriptor_.holovibes_.get_cd().current_window);
        // Floating number issue: cast to float for the comparison
        const float val = value;
        if (old_val != val)
        {
            ui_descriptor_.holovibes_.get_cd().set_contrast_max(ui_descriptor_.holovibes_.get_cd().current_window,
                                                                value);
            pipe_refresh();
        }
    }
}

void MainWindow::invert_contrast(bool value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    if (ui_descriptor_.holovibes_.get_cd().contrast_enabled)
    {
        ui_descriptor_.holovibes_.get_cd().contrast_invert = value;
        pipe_refresh();
    }
}

void MainWindow::set_auto_refresh_contrast(bool value)
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().contrast_auto_refresh = value;
    pipe_refresh();
    notify();
}

void MainWindow::set_log_scale(const bool value)
{
    LOG_INFO;
    if (::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_))
        return;

    ui_descriptor_.holovibes_.get_cd().set_log_scale_slice_enabled(ui_descriptor_.holovibes_.get_cd().current_window,
                                                                   value);
    if (value && ui_descriptor_.holovibes_.get_cd().contrast_enabled)
        set_auto_contrast();
    pipe_refresh();
    notify();
}
#pragma endregion
/* ------------ */
#pragma region Convolution
void MainWindow::update_convo_kernel(const QString& value)
{
    LOG_INFO;
    if (ui_descriptor_.holovibes_.get_cd().convolution_enabled)
    {
        ui_descriptor_.holovibes_.get_cd().set_convolution(true,
                                                           ui.KernelQuickSelectComboBox->currentText().toStdString());

        try
        {
            auto pipe = ui_descriptor_.holovibes_.get_compute_pipe();
            pipe->request_convolution();
            // Wait for the convolution to be enabled for notify
            while (pipe->get_convolution_requested())
                continue;
        }
        catch (const std::exception& e)
        {
            ui_descriptor_.holovibes_.get_cd().convolution_enabled = false;
            LOG_ERROR << e.what();
        }

        notify();
    }
}

void MainWindow::set_convolution_mode(const bool value)
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().set_convolution(value,
                                                       ui.KernelQuickSelectComboBox->currentText().toStdString());

    ::holovibes::api::set_convolution_mode(ui_descriptor_, value);

    notify();
}

void MainWindow::set_divide_convolution_mode(const bool value)
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().divide_convolution_enabled = value;

    pipe_refresh();
    notify();
}

void MainWindow::set_fast_pipe(bool value)
{
    LOG_INFO;
    auto pipe = dynamic_cast<Pipe*>(ui_descriptor_.holovibes_.get_compute_pipe().get());
    if (value && pipe)
    {
        pipe->insert_fn_end_vect([=]() {
            // Constraints linked with fast pipe option
            ui_descriptor_.holovibes_.get_cd().time_transformation_stride =
                ui_descriptor_.holovibes_.get_cd().batch_size.load();
            ui_descriptor_.holovibes_.get_cd().time_transformation_size =
                ui_descriptor_.holovibes_.get_cd().batch_size.load();
            pipe->request_update_time_transformation_stride();
            pipe->request_update_time_transformation_size();
            ui_descriptor_.holovibes_.get_cd().fast_pipe = true;
            pipe_refresh();
            notify();
        });
    }
    else
    {
        ui_descriptor_.holovibes_.get_cd().fast_pipe = false;
        pipe_refresh();
        notify();
    }
}

#pragma endregion
/* ------------ */
#pragma region Reticle
void MainWindow::display_reticle(bool value)
{
    LOG_INFO;
    ui_descriptor_.holovibes_.get_cd().reticle_enabled = value;
    if (value)
    {
        ui_descriptor_.mainDisplay->getOverlayManager().create_overlay<Reticle>();
        ui_descriptor_.mainDisplay->getOverlayManager().create_default();
    }
    else
    {
        ui_descriptor_.mainDisplay->getOverlayManager().disable_all(Reticle);
    }
    pipe_refresh();
    notify();
}

void MainWindow::reticle_scale(double value)
{
    LOG_INFO;
    if (0 > value || value > 1)
        return;

    ui_descriptor_.holovibes_.get_cd().reticle_scale = value;
    pipe_refresh();
}
#pragma endregion Reticle
/* ------------ */
#pragma region Chart
void MainWindow::activeSignalZone()
{
    LOG_INFO;
    ui_descriptor_.mainDisplay->getOverlayManager().create_overlay<Signal>();
    notify();
}

void MainWindow::activeNoiseZone()
{
    LOG_INFO;
    ui_descriptor_.mainDisplay->getOverlayManager().create_overlay<Noise>();
    notify();
}

void MainWindow::start_chart_display()
{
    LOG_INFO;
    if (ui_descriptor_.holovibes_.get_cd().chart_display_enabled)
        return;

    auto pipe = ui_descriptor_.holovibes_.get_compute_pipe();
    pipe->request_display_chart();

    // Wait for the chart display to be enabled for notify
    while (pipe->get_chart_display_requested())
        continue;

    ui_descriptor_.plot_window_ =
        std::make_unique<PlotWindow>(*ui_descriptor_.holovibes_.get_compute_pipe()->get_chart_display_queue(),
                                     ui_descriptor_.auto_scale_point_threshold_,
                                     "Chart");
    connect(ui_descriptor_.plot_window_.get(),
            SIGNAL(closed()),
            this,
            SLOT(stop_chart_display()),
            Qt::UniqueConnection);

    ui.ChartPlotPushButton->setEnabled(false);
}

void MainWindow::stop_chart_display()
{
    LOG_INFO;
    if (!ui_descriptor_.holovibes_.get_cd().chart_display_enabled)
        return;

    try
    {
        auto pipe = ui_descriptor_.holovibes_.get_compute_pipe();
        pipe->request_disable_display_chart();

        // Wait for the chart display to be disabled for notify
        while (pipe->get_disable_chart_display_requested())
            continue;
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << e.what();
    }

    ui_descriptor_.plot_window_.reset(nullptr);

    ui.ChartPlotPushButton->setEnabled(true);
}
#pragma endregion
/* ------------ */
#pragma region Record
void MainWindow::set_record_frame_step(int value)
{
    LOG_INFO;
    ::holovibes::api::set_record_frame_step(ui_descriptor_, value);
    ui.NumberOfFramesSpinBox->setSingleStep(value);
}

void MainWindow::set_nb_frames_mode(bool value)
{
    LOG_INFO;
    ui.NumberOfFramesSpinBox->setEnabled(value);
}

void MainWindow::browse_record_output_file()
{
    LOG_INFO;
    QString filepath;

    // Open file explorer dialog on the fly depending on the record mode
    // Add the matched extension to the file if none
    if (ui_descriptor_.record_mode_ == RecordMode::CHART)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Chart output file"),
                                                ui_descriptor_.record_output_directory_.c_str(),
                                                tr("Text files (*.txt);;CSV files (*.csv)"));
    }
    else if (ui_descriptor_.record_mode_ == RecordMode::RAW)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                ui_descriptor_.record_output_directory_.c_str(),
                                                tr("Holo files (*.holo)"));
    }
    else if (ui_descriptor_.record_mode_ == RecordMode::HOLOGRAM)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                ui_descriptor_.record_output_directory_.c_str(),
                                                tr("Holo files (*.holo);; Avi Files (*.avi);; Mp4 files (*.mp4)"));
    }

    if (filepath.isEmpty())
        return;

    // Convert QString to std::string
    std::string std_filepath = filepath.toStdString();

    const std::string file_ext = ::holovibes::api::browse_record_output_file(std_filepath,
                                                                             ui_descriptor_.record_output_directory_,
                                                                             ui_descriptor_.default_output_filename_);

    // Will pick the item combobox related to file_ext if it exists, else, nothing is done
    ui.RecordExtComboBox->setCurrentText(file_ext.c_str());

    notify();
}

void MainWindow::browse_batch_input()
{
    LOG_INFO;

    // Open file explorer on the fly
    QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("Batch input file"),
                                                    ui_descriptor_.batch_input_directory_.c_str(),
                                                    tr("All files (*)"));

    // Output the file selected in he ui line edit widget
    QLineEdit* batch_input_line_edit = ui.BatchInputPathLineEdit;
    batch_input_line_edit->clear();
    batch_input_line_edit->insert(filename);
}

void MainWindow::set_record_mode(const QString& value)
{
    LOG_INFO;
    if (ui_descriptor_.record_mode_ == RecordMode::CHART)
        stop_chart_display();

    ::holovibes::api::stop_record(ui_descriptor_.holovibes_, ui_descriptor_.record_mode_);

    const std::string text = value.toStdString();
    ::holovibes::api::set_record_mode(text, ui_descriptor_.record_mode_);

    if (ui_descriptor_.record_mode_ == RecordMode::CHART)
    {
        ui.RecordExtComboBox->clear();
        ui.RecordExtComboBox->insertItem(0, ".csv");
        ui.RecordExtComboBox->insertItem(1, ".txt");

        ui.ChartPlotWidget->show();

        if (ui_descriptor_.mainDisplay)
        {
            ui_descriptor_.mainDisplay->resetTransform();

            ui_descriptor_.mainDisplay->getOverlayManager().enable_all(Signal);
            ui_descriptor_.mainDisplay->getOverlayManager().enable_all(Noise);
            ui_descriptor_.mainDisplay->getOverlayManager().create_overlay<Signal>();
        }
    }
    else
    {
        if (ui_descriptor_.record_mode_ == RecordMode::RAW)
        {
            ui.RecordExtComboBox->clear();
            ui.RecordExtComboBox->insertItem(0, ".holo");
        }
        else if (ui_descriptor_.record_mode_ == RecordMode::HOLOGRAM)
        {
            ui.RecordExtComboBox->clear();
            ui.RecordExtComboBox->insertItem(0, ".holo");
            ui.RecordExtComboBox->insertItem(1, ".avi");
            ui.RecordExtComboBox->insertItem(2, ".mp4");
        }

        ui.ChartPlotWidget->hide();

        if (ui_descriptor_.mainDisplay)
        {
            ui_descriptor_.mainDisplay->resetTransform();

            ui_descriptor_.mainDisplay->getOverlayManager().disable_all(Signal);
            ui_descriptor_.mainDisplay->getOverlayManager().disable_all(Noise);
        }
    }

    notify();
}

void MainWindow::stop_record()
{
    LOG_INFO;
    ::holovibes::api::stop_record(ui_descriptor_.holovibes_, ui_descriptor_.record_mode_);
}

void MainWindow::record_finished(RecordMode record_mode)
{
    LOG_INFO;
    std::string info;

    if (record_mode == RecordMode::CHART)
        info = "Chart record finished";
    else if (record_mode == RecordMode::HOLOGRAM || record_mode == RecordMode::RAW)
        info = "Frame record finished";

    ui.RecordProgressBar->hide();

    if (ui.BatchGroupBox->isChecked())
        info = "Batch " + info;

    LOG_INFO << "[RECORDER] " << info;

    ui.RawDisplayingCheckBox->setHidden(false);
    ui.ExportRecPushButton->setEnabled(true);
    ui.ExportStopPushButton->setEnabled(false);
    ui.BatchSizeSpinBox->setEnabled(ui_descriptor_.holovibes_.get_cd().compute_mode == Computation::Hologram);
    ui_descriptor_.is_recording_ = false;
}

void MainWindow::start_record()
{
    LOG_INFO;
    bool batch_enabled = ui.BatchGroupBox->isChecked();
    bool nb_frame_checked = ui.NumberOfFramesCheckBox->isChecked();
    std::optional<unsigned int> nb_frames_to_record = std::nullopt;
    if (nb_frame_checked)
    {
        nb_frames_to_record = ui.NumberOfFramesSpinBox->value();
    }

    std::string batch_input_path = ui.BatchInputPathLineEdit->text().toUtf8();

    // Preconditions to start record
    const bool preconditions = ::holovibes::api::start_record_preconditions(ui_descriptor_,
                                                                            batch_enabled,
                                                                            nb_frame_checked,
                                                                            nb_frames_to_record,
                                                                            batch_input_path);

    if (!preconditions)
    {
        return;
    }

    std::string output_path =
        ui.OutputFilePathLineEdit->text().toStdString() + ui.RecordExtComboBox->currentText().toStdString();

    // Start record
    ui_descriptor_.raw_window.reset(nullptr);
    disable_raw_view();
    ui.RawDisplayingCheckBox->setHidden(true);

    ui.BatchSizeSpinBox->setEnabled(false);
    ui_descriptor_.is_recording_ = true;

    ui.ExportRecPushButton->setEnabled(false);
    ui.ExportStopPushButton->setEnabled(true);

    ui.RecordProgressBar->reset();
    ui.RecordProgressBar->show();

    auto callback = [record_mode = ui_descriptor_.record_mode_, this]() {
        synchronize_thread([=]() { record_finished(record_mode); });
    };

    ::holovibes::api::start_record(ui_descriptor_,
                                   batch_enabled,
                                   nb_frames_to_record,
                                   output_path,
                                   batch_input_path,
                                   callback);
}
#pragma endregion
/* ------------ */
#pragma region Import
void MainWindow::set_start_stop_buttons(bool value)
{
    LOG_INFO;
    ui.ImportStartPushButton->setEnabled(value);
    ui.ImportStopPushButton->setEnabled(value);
}

void MainWindow::import_browse_file()
{
    LOG_INFO;
    QString filename = "";
    // Open the file explorer to let the user pick his file
    // and store the chosen file in filename

    filename = QFileDialog::getOpenFileName(this,
                                            tr("import file"),
                                            ui_descriptor_.file_input_directory_.c_str(),
                                            tr("All files (*.holo *.cine);; Holo files (*.holo);; Cine files "
                                               "(*.cine)"));

    // Get the widget (output bar) from the ui linked to the file explorer
    QLineEdit* import_line_edit = ui.ImportPathLineEdit;
    // Insert the newly getted path in it
    import_line_edit->clear();
    import_line_edit->insert(filename);

    // Start importing the chosen
    std::optional<io_files::InputFrameFile*> input_file_opt;
    try
    {
        input_file_opt = ::holovibes::api::import_file(filename.toStdString());
    }
    catch (const io_files::FileException& e)
    {
        // In case of bad format, we triggered the user
        QMessageBox messageBox;
        messageBox.critical(nullptr, "File Error", e.what());
        LOG_ERROR << e.what();

        // Holovibes cannot be launched over this file
        set_start_stop_buttons(false);
        return;
    }

    if (input_file_opt)
    {
        auto input_file = input_file_opt.value();
        // Gather data from the newly opened file
        size_t nb_frames = input_file->get_total_nb_frames();
        ui_descriptor_.file_fd_ = input_file->get_frame_descriptor();
        input_file->import_compute_settings(ui_descriptor_.holovibes_.get_cd());

        // Don't need the input file anymore
        delete input_file;

        // Update the ui with the gathered data
        ui.ImportEndIndexSpinBox->setMaximum(nb_frames);
        ui.ImportEndIndexSpinBox->setValue(nb_frames);

        // We can now launch holovibes over this file
        set_start_stop_buttons(true);
    }
    else
        set_start_stop_buttons(false);
}

void MainWindow::import_stop()
{
    LOG_INFO;
    ::holovibes::api::close_windows(ui_descriptor_.holovibes_,
                                    ui_descriptor_.mainDisplay,
                                    ui_descriptor_.sliceXZ,
                                    ui_descriptor_.sliceYZ,
                                    ui_descriptor_.lens_window,
                                    ui_descriptor_.raw_window,
                                    ui_descriptor_.filter2d_window,
                                    ui_descriptor_.plot_window_);
    cancel_time_transformation_cuts();

    ::holovibes::api::import_stop(ui_descriptor_);
    synchronize_thread([&]() { ui.FileReaderProgressBar->hide(); });
    notify();
}

void MainWindow::import_start()
{
    LOG_INFO;
    // shift main window when camera view appears
    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();
    move(QPoint(210 + (screen_width - 800) / 2, 200 + (screen_height - 500) / 2));

    QLineEdit* import_line_edit = ui.ImportPathLineEdit;
    QSpinBox* fps_spinbox = ui.ImportInputFpsSpinBox;
    QSpinBox* start_spinbox = ui.ImportStartIndexSpinBox;
    QCheckBox* load_file_gpu_box = ui.LoadFileInGpuCheckBox;
    QSpinBox* end_spinbox = ui.ImportEndIndexSpinBox;

    bool res_import_start = ::holovibes::api::import_start(ui_descriptor_,
                                                           import_line_edit->text().toStdString(),
                                                           fps_spinbox->value(),
                                                           start_spinbox->value(),
                                                           load_file_gpu_box->isChecked(),
                                                           end_spinbox->value());

    if (res_import_start)
    {
        ui.FileReaderProgressBar->show();
        ui_descriptor_.is_enabled_camera_ = true;
        set_image_mode(nullptr);

        // Make camera's settings menu unaccessible
        QAction* settings = ui.actionSettings;
        settings->setEnabled(false);

        ui_descriptor_.import_type_ = ::holovibes::UserInterfaceDescriptor::ImportType::File;

        notify();
    }
    else
    {
        ui_descriptor_.mainDisplay.reset(nullptr);
    }

    ui.ImageModeComboBox->setCurrentIndex(::holovibes::api::is_raw_mode(ui_descriptor_.holovibes_) ? 0 : 1);
}

void MainWindow::import_start_spinbox_update()
{
    LOG_INFO;
    QSpinBox* start_spinbox = ui.ImportStartIndexSpinBox;
    QSpinBox* end_spinbox = ui.ImportEndIndexSpinBox;

    if (start_spinbox->value() > end_spinbox->value())
        end_spinbox->setValue(start_spinbox->value());
}

void MainWindow::import_end_spinbox_update()
{
    LOG_INFO;
    QSpinBox* start_spinbox = ui.ImportStartIndexSpinBox;
    QSpinBox* end_spinbox = ui.ImportEndIndexSpinBox;

    if (end_spinbox->value() < start_spinbox->value())
        start_spinbox->setValue(end_spinbox->value());
}

#pragma endregion

#pragma region Themes
void MainWindow::set_night()
{
    LOG_INFO;
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
    ui_descriptor_.theme_index_ = 1;
}

void MainWindow::set_classic()
{
    LOG_INFO;
    qApp->setPalette(this->style()->standardPalette());
    // Light mode style
    qApp->setStyle(QStyleFactory::create("WindowsVista"));
    qApp->setStyleSheet("");
    ui_descriptor_.theme_index_ = 0;
}
#pragma endregion

#pragma region Getters

RawWindow* MainWindow::get_main_display()
{
    LOG_INFO;
    return ui_descriptor_.mainDisplay.get();
}

void MainWindow::update_file_reader_index(int n)
{
    LOG_INFO;
    auto lambda = [this, n]() { ui.FileReaderProgressBar->setValue(n); };
    synchronize_thread(lambda);
}
#pragma endregion
} // namespace gui
} // namespace holovibes
