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

    set_z_step(UserInterfaceDescriptor::instance().z_step_);
    set_record_frame_step(UserInterfaceDescriptor::instance().record_frame_step_);
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
    Holovibes::instance().get_cd().compute_mode = Computation::Raw;
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

    ::holovibes::api::close_windows();
    ::holovibes::api::close_critical_compute();
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
    ui.InputBrowseToolButton->setEnabled(Holovibes::instance().get_cd().is_computation_stopped);

    // Tabs
    if (Holovibes::instance().get_cd().is_computation_stopped)
    {
        ui.CompositeGroupBox->hide();
        ui.ImageRenderingGroupBox->setEnabled(false);
        ui.ViewGroupBox->setEnabled(false);
        ui.ExportGroupBox->setEnabled(false);
        layout_toggled();
        return;
    }

    if (UserInterfaceDescriptor::instance().is_enabled_camera_ &&
        Holovibes::instance().get_cd().compute_mode == Computation::Raw)
    {
        ui.ImageRenderingGroupBox->setEnabled(true);
        ui.ViewGroupBox->setEnabled(false);
        ui.ExportGroupBox->setEnabled(true);
    }

    else if (UserInterfaceDescriptor::instance().is_enabled_camera_ &&
             Holovibes::instance().get_cd().compute_mode == Computation::Hologram)
    {
        ui.ImageRenderingGroupBox->setEnabled(true);
        ui.ViewGroupBox->setEnabled(true);
        ui.ExportGroupBox->setEnabled(true);
    }

    const bool is_raw = ::holovibes::api::is_raw_mode();

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
    ui.RawDisplayingCheckBox->setChecked(!is_raw && Holovibes::instance().get_cd().raw_view_enabled);

    QPushButton* signalBtn = ui.ChartSignalPushButton;
    signalBtn->setStyleSheet(
        (UserInterfaceDescriptor::instance().mainDisplay && signalBtn->isEnabled() &&
         UserInterfaceDescriptor::instance().mainDisplay->getKindOfOverlay() == KindOfOverlay::Signal)
            ? "QPushButton {color: #8E66D9;}"
            : "");

    QPushButton* noiseBtn = ui.ChartNoisePushButton;
    noiseBtn->setStyleSheet(
        (UserInterfaceDescriptor::instance().mainDisplay && noiseBtn->isEnabled() &&
         UserInterfaceDescriptor::instance().mainDisplay->getKindOfOverlay() == KindOfOverlay::Noise)
            ? "QPushButton {color: #00A4AB;}"
            : "");

    ui.PhaseUnwrap2DCheckBox->setEnabled(Holovibes::instance().get_cd().img_type == ImgType::PhaseIncrease ||
                                         Holovibes::instance().get_cd().img_type == ImgType::Argument);

    // Time transformation cuts
    ui.TimeTransformationCutsCheckBox->setChecked(!is_raw &&
                                                  Holovibes::instance().get_cd().time_transformation_cuts_enabled);

    // Contrast
    ui.ContrastCheckBox->setChecked(!is_raw && Holovibes::instance().get_cd().contrast_enabled);
    ui.ContrastCheckBox->setEnabled(true);
    ui.AutoRefreshContrastCheckBox->setChecked(Holovibes::instance().get_cd().contrast_auto_refresh);

    // Contrast SpinBox:
    ui.ContrastMinDoubleSpinBox->setEnabled(!Holovibes::instance().get_cd().contrast_auto_refresh);
    ui.ContrastMinDoubleSpinBox->setValue(
        Holovibes::instance().get_cd().get_contrast_min(Holovibes::instance().get_cd().current_window));
    ui.ContrastMaxDoubleSpinBox->setEnabled(!Holovibes::instance().get_cd().contrast_auto_refresh);
    ui.ContrastMaxDoubleSpinBox->setValue(
        Holovibes::instance().get_cd().get_contrast_max(Holovibes::instance().get_cd().current_window));

    // FFT shift
    ui.FFTShiftCheckBox->setChecked(Holovibes::instance().get_cd().fft_shift_enabled);
    ui.FFTShiftCheckBox->setEnabled(true);

    // Window selection
    QComboBox* window_selection = ui.WindowSelectionComboBox;
    window_selection->setEnabled(Holovibes::instance().get_cd().time_transformation_cuts_enabled);
    window_selection->setCurrentIndex(
        window_selection->isEnabled() ? static_cast<int>(Holovibes::instance().get_cd().current_window.load()) : 0);

    ui.LogScaleCheckBox->setEnabled(true);
    ui.LogScaleCheckBox->setChecked(!is_raw && Holovibes::instance().get_cd().get_img_log_scale_slice_enabled(
                                                   Holovibes::instance().get_cd().current_window.load()));
    ui.ImgAccuCheckBox->setEnabled(true);
    ui.ImgAccuCheckBox->setChecked(!is_raw && Holovibes::instance().get_cd().get_img_acc_slice_enabled(
                                                  Holovibes::instance().get_cd().current_window.load()));
    ui.ImgAccuSpinBox->setValue(
        Holovibes::instance().get_cd().get_img_acc_slice_level(Holovibes::instance().get_cd().current_window.load()));
    if (Holovibes::instance().get_cd().current_window == WindowKind::XYview)
    {
        ui.RotatePushButton->setText(
            ("Rot " + std::to_string(static_cast<int>(UserInterfaceDescriptor::instance().displayAngle))).c_str());
        ui.FlipPushButton->setText(("Flip " + std::to_string(UserInterfaceDescriptor::instance().displayFlip)).c_str());
    }
    else if (Holovibes::instance().get_cd().current_window == WindowKind::XZview)
    {
        ui.RotatePushButton->setText(
            ("Rot " + std::to_string(static_cast<int>(UserInterfaceDescriptor::instance().xzAngle))).c_str());
        ui.FlipPushButton->setText(("Flip " + std::to_string(UserInterfaceDescriptor::instance().xzFlip)).c_str());
    }
    else if (Holovibes::instance().get_cd().current_window == WindowKind::YZview)
    {
        ui.RotatePushButton->setText(
            ("Rot " + std::to_string(static_cast<int>(UserInterfaceDescriptor::instance().yzAngle))).c_str());
        ui.FlipPushButton->setText(("Flip " + std::to_string(UserInterfaceDescriptor::instance().yzFlip)).c_str());
    }

    // p accu
    ui.PAccuCheckBox->setEnabled(Holovibes::instance().get_cd().img_type != ImgType::PhaseIncrease);
    ui.PAccuCheckBox->setChecked(Holovibes::instance().get_cd().p_accu_enabled);
    ui.PAccSpinBox->setMaximum(Holovibes::instance().get_cd().time_transformation_size - 1);
    if (Holovibes::instance().get_cd().p_acc_level > Holovibes::instance().get_cd().time_transformation_size - 1)
        Holovibes::instance().get_cd().p_acc_level = Holovibes::instance().get_cd().time_transformation_size - 1;
    ui.PAccSpinBox->setValue(Holovibes::instance().get_cd().p_acc_level);
    ui.PAccSpinBox->setEnabled(Holovibes::instance().get_cd().img_type != ImgType::PhaseIncrease);
    if (Holovibes::instance().get_cd().p_accu_enabled)
    {
        ui.PSpinBox->setMaximum(Holovibes::instance().get_cd().time_transformation_size -
                                Holovibes::instance().get_cd().p_acc_level - 1);
        if (Holovibes::instance().get_cd().pindex >
            Holovibes::instance().get_cd().time_transformation_size - Holovibes::instance().get_cd().p_acc_level - 1)
            Holovibes::instance().get_cd().pindex = Holovibes::instance().get_cd().time_transformation_size -
                                                    Holovibes::instance().get_cd().p_acc_level - 1;
        ui.PSpinBox->setValue(Holovibes::instance().get_cd().pindex);
        ui.PAccSpinBox->setMaximum(Holovibes::instance().get_cd().time_transformation_size -
                                   Holovibes::instance().get_cd().pindex - 1);
    }
    else
    {
        ui.PSpinBox->setMaximum(Holovibes::instance().get_cd().time_transformation_size - 1);
        if (Holovibes::instance().get_cd().pindex > Holovibes::instance().get_cd().time_transformation_size - 1)
            Holovibes::instance().get_cd().pindex = Holovibes::instance().get_cd().time_transformation_size - 1;
        ui.PSpinBox->setValue(Holovibes::instance().get_cd().pindex);
    }
    ui.PSpinBox->setEnabled(!is_raw);

    // q accu
    bool is_ssa_stft = Holovibes::instance().get_cd().time_transformation == TimeTransformation::SSA_STFT;
    ui.Q_AccuCheckBox->setEnabled(is_ssa_stft && !is_raw);
    ui.Q_AccSpinBox->setEnabled(is_ssa_stft && !is_raw);
    ui.Q_SpinBox->setEnabled(is_ssa_stft && !is_raw);

    ui.Q_AccuCheckBox->setChecked(Holovibes::instance().get_cd().q_acc_enabled);
    ui.Q_AccSpinBox->setMaximum(Holovibes::instance().get_cd().time_transformation_size - 1);
    if (Holovibes::instance().get_cd().q_acc_level > Holovibes::instance().get_cd().time_transformation_size - 1)
        Holovibes::instance().get_cd().q_acc_level = Holovibes::instance().get_cd().time_transformation_size - 1;
    ui.Q_AccSpinBox->setValue(Holovibes::instance().get_cd().q_acc_level);
    if (Holovibes::instance().get_cd().q_acc_enabled)
    {
        ui.Q_SpinBox->setMaximum(Holovibes::instance().get_cd().time_transformation_size -
                                 Holovibes::instance().get_cd().q_acc_level - 1);
        if (Holovibes::instance().get_cd().q_index >
            Holovibes::instance().get_cd().time_transformation_size - Holovibes::instance().get_cd().q_acc_level - 1)
            Holovibes::instance().get_cd().q_index = Holovibes::instance().get_cd().time_transformation_size -
                                                     Holovibes::instance().get_cd().q_acc_level - 1;
        ui.Q_SpinBox->setValue(Holovibes::instance().get_cd().q_index);
        ui.Q_AccSpinBox->setMaximum(Holovibes::instance().get_cd().time_transformation_size -
                                    Holovibes::instance().get_cd().q_index - 1);
    }
    else
    {
        ui.Q_SpinBox->setMaximum(Holovibes::instance().get_cd().time_transformation_size - 1);
        if (Holovibes::instance().get_cd().q_index > Holovibes::instance().get_cd().time_transformation_size - 1)
            Holovibes::instance().get_cd().q_index = Holovibes::instance().get_cd().time_transformation_size - 1;
        ui.Q_SpinBox->setValue(Holovibes::instance().get_cd().q_index);
    }

    // XY accu
    ui.XAccuCheckBox->setChecked(Holovibes::instance().get_cd().x_accu_enabled);
    ui.XAccSpinBox->setValue(Holovibes::instance().get_cd().x_acc_level);
    ui.YAccuCheckBox->setChecked(Holovibes::instance().get_cd().y_accu_enabled);
    ui.YAccSpinBox->setValue(Holovibes::instance().get_cd().y_acc_level);

    int max_width = 0;
    int max_height = 0;
    if (Holovibes::instance().get_gpu_input_queue() != nullptr)
    {
        max_width = Holovibes::instance().get_gpu_input_queue()->get_fd().width - 1;
        max_height = Holovibes::instance().get_gpu_input_queue()->get_fd().height - 1;
    }
    else
    {
        Holovibes::instance().get_cd().x_cuts = 0;
        Holovibes::instance().get_cd().y_cuts = 0;
    }
    ui.XSpinBox->setMaximum(max_width);
    ui.YSpinBox->setMaximum(max_height);
    QSpinBoxQuietSetValue(ui.XSpinBox, Holovibes::instance().get_cd().x_cuts);
    QSpinBoxQuietSetValue(ui.YSpinBox, Holovibes::instance().get_cd().y_cuts);

    // Time transformation
    ui.TimeTransformationStrideSpinBox->setEnabled(!is_raw);

    const uint input_queue_capacity = global::global_config.input_queue_max_size;

    ui.TimeTransformationStrideSpinBox->setValue(Holovibes::instance().get_cd().time_transformation_stride);
    ui.TimeTransformationStrideSpinBox->setSingleStep(Holovibes::instance().get_cd().batch_size);
    ui.TimeTransformationStrideSpinBox->setMinimum(Holovibes::instance().get_cd().batch_size);

    // Batch
    ui.BatchSizeSpinBox->setEnabled(!is_raw && !UserInterfaceDescriptor::instance().is_recording_);

    if (Holovibes::instance().get_cd().batch_size > input_queue_capacity)
        Holovibes::instance().get_cd().batch_size = input_queue_capacity;

    ui.BatchSizeSpinBox->setValue(Holovibes::instance().get_cd().batch_size);
    ui.BatchSizeSpinBox->setMaximum(input_queue_capacity);

    // Image rendering
    ui.SpaceTransformationComboBox->setEnabled(!is_raw &&
                                               !Holovibes::instance().get_cd().time_transformation_cuts_enabled);
    ui.SpaceTransformationComboBox->setCurrentIndex(
        static_cast<int>(Holovibes::instance().get_cd().space_transformation.load()));
    ui.TimeTransformationComboBox->setEnabled(!is_raw);
    ui.TimeTransformationComboBox->setCurrentIndex(
        static_cast<int>(Holovibes::instance().get_cd().time_transformation.load()));

    // Changing time_transformation_size with time transformation cuts is
    // supported by the pipe, but some modifications have to be done in
    // SliceWindow, OpenGl buffers.
    ui.timeTransformationSizeSpinBox->setEnabled(!is_raw &&
                                                 !Holovibes::instance().get_cd().time_transformation_cuts_enabled);
    ui.timeTransformationSizeSpinBox->setValue(Holovibes::instance().get_cd().time_transformation_size);
    ui.TimeTransformationCutsCheckBox->setEnabled(ui.timeTransformationSizeSpinBox->value() >=
                                                  MIN_IMG_NB_TIME_TRANSFORMATION_CUTS);

    ui.WaveLengthDoubleSpinBox->setEnabled(!is_raw);
    ui.WaveLengthDoubleSpinBox->setValue(Holovibes::instance().get_cd().lambda * 1.0e9f);
    ui.ZDoubleSpinBox->setEnabled(!is_raw);
    ui.ZDoubleSpinBox->setValue(Holovibes::instance().get_cd().zdistance);
    ui.BoundaryLineEdit->setText(QString::number(Holovibes::instance().get_boundary()));

    // Filter2d
    ui.Filter2D->setEnabled(!is_raw);
    ui.Filter2D->setChecked(!is_raw && Holovibes::instance().get_cd().filter2d_enabled);
    ui.Filter2DView->setEnabled(!is_raw && Holovibes::instance().get_cd().filter2d_enabled);
    ui.Filter2DView->setChecked(!is_raw && Holovibes::instance().get_cd().filter2d_view_enabled);
    ui.Filter2DN1SpinBox->setEnabled(!is_raw && Holovibes::instance().get_cd().filter2d_enabled);
    ui.Filter2DN1SpinBox->setValue(Holovibes::instance().get_cd().filter2d_n1);
    ui.Filter2DN1SpinBox->setMaximum(ui.Filter2DN2SpinBox->value() - 1);
    ui.Filter2DN2SpinBox->setEnabled(!is_raw && Holovibes::instance().get_cd().filter2d_enabled);
    ui.Filter2DN2SpinBox->setValue(Holovibes::instance().get_cd().filter2d_n2);

    // Composite
    const int time_transformation_size_max = Holovibes::instance().get_cd().time_transformation_size - 1;
    ui.PRedSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui.PBlueSpinBox_Composite->setMaximum(time_transformation_size_max);
    ui.SpinBox_hue_freq_min->setMaximum(time_transformation_size_max);
    ui.SpinBox_hue_freq_max->setMaximum(time_transformation_size_max);
    ui.SpinBox_saturation_freq_min->setMaximum(time_transformation_size_max);
    ui.SpinBox_saturation_freq_max->setMaximum(time_transformation_size_max);
    ui.SpinBox_value_freq_min->setMaximum(time_transformation_size_max);
    ui.SpinBox_value_freq_max->setMaximum(time_transformation_size_max);

    ui.RenormalizationCheckBox->setChecked(Holovibes::instance().get_cd().composite_auto_weights_);

    QSpinBoxQuietSetValue(ui.PRedSpinBox_Composite, Holovibes::instance().get_cd().composite_p_red);
    QSpinBoxQuietSetValue(ui.PBlueSpinBox_Composite, Holovibes::instance().get_cd().composite_p_blue);
    QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_R, Holovibes::instance().get_cd().weight_r);
    QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_G, Holovibes::instance().get_cd().weight_g);
    QDoubleSpinBoxQuietSetValue(ui.WeightSpinBox_B, Holovibes::instance().get_cd().weight_b);
    actualize_frequency_channel_v();

    QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_min, Holovibes::instance().get_cd().composite_p_min_h);
    QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_max, Holovibes::instance().get_cd().composite_p_max_h);
    QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_min,
                         (int)(Holovibes::instance().get_cd().slider_h_threshold_min * 1000));
    slide_update_threshold_h_min();
    QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_max,
                         (int)(Holovibes::instance().get_cd().slider_h_threshold_max * 1000));
    slide_update_threshold_h_max();

    QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_min, Holovibes::instance().get_cd().composite_p_min_s);
    QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_max, Holovibes::instance().get_cd().composite_p_max_s);
    QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_min,
                         (int)(Holovibes::instance().get_cd().slider_s_threshold_min * 1000));
    slide_update_threshold_s_min();
    QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_max,
                         (int)(Holovibes::instance().get_cd().slider_s_threshold_max * 1000));
    slide_update_threshold_s_max();

    QSpinBoxQuietSetValue(ui.SpinBox_value_freq_min, Holovibes::instance().get_cd().composite_p_min_v);
    QSpinBoxQuietSetValue(ui.SpinBox_value_freq_max, Holovibes::instance().get_cd().composite_p_max_v);
    QSliderQuietSetValue(ui.horizontalSlider_value_threshold_min,
                         (int)(Holovibes::instance().get_cd().slider_v_threshold_min * 1000));
    slide_update_threshold_v_min();
    QSliderQuietSetValue(ui.horizontalSlider_value_threshold_max,
                         (int)(Holovibes::instance().get_cd().slider_v_threshold_max * 1000));
    slide_update_threshold_v_max();

    ui.CompositeGroupBox->setHidden(::holovibes::api::is_raw_mode() ||
                                    (Holovibes::instance().get_cd().img_type != ImgType::Composite));

    bool rgbMode = ui.radioButton_rgb->isChecked();
    ui.groupBox->setHidden(!rgbMode);
    ui.groupBox_5->setHidden(!rgbMode && !ui.RenormalizationCheckBox->isChecked());
    ui.groupBox_hue->setHidden(rgbMode);
    ui.groupBox_saturation->setHidden(rgbMode);
    ui.groupBox_value->setHidden(rgbMode);

    // Reticle
    ui.ReticleScaleDoubleSpinBox->setEnabled(Holovibes::instance().get_cd().reticle_enabled);
    ui.ReticleScaleDoubleSpinBox->setValue(Holovibes::instance().get_cd().reticle_scale);
    ui.DisplayReticleCheckBox->setChecked(Holovibes::instance().get_cd().reticle_enabled);

    // Lens View
    ui.LensViewCheckBox->setChecked(Holovibes::instance().get_cd().gpu_lens_display_enabled);

    // Renormalize
    ui.RenormalizeCheckBox->setChecked(Holovibes::instance().get_cd().renorm_enabled);

    // Convolution
    ui.ConvoCheckBox->setEnabled(Holovibes::instance().get_cd().compute_mode == Computation::Hologram);
    ui.ConvoCheckBox->setChecked(Holovibes::instance().get_cd().convolution_enabled);
    ui.DivideConvoCheckBox->setChecked(Holovibes::instance().get_cd().convolution_enabled &&
                                       Holovibes::instance().get_cd().divide_convolution_enabled);

    QLineEdit* path_line_edit = ui.OutputFilePathLineEdit;
    path_line_edit->clear();

    std::string record_output_path =
        (std::filesystem::path(UserInterfaceDescriptor::instance().record_output_directory_) /
         UserInterfaceDescriptor::instance().default_output_filename_)
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
                Holovibes::instance().get_cd().pindex = 0;
                Holovibes::instance().get_cd().time_transformation_size = 1;
                if (Holovibes::instance().get_cd().convolution_enabled)
                {
                    Holovibes::instance().get_cd().convolution_enabled = false;
                }
                ::holovibes::api::close_windows();
                ::holovibes::api::close_critical_compute();
                LOG_ERROR << "GPU computing error occured.";
                notify();
            };
            synchronize_thread(lambda);
        }

        auto lambda = [this, accu = (dynamic_cast<const AccumulationException*>(err_ptr) != nullptr)] {
            if (accu)
            {
                Holovibes::instance().get_cd().img_acc_slice_xy_enabled = false;
                Holovibes::instance().get_cd().img_acc_slice_xy_level = 1;
            }
            ::holovibes::api::close_critical_compute();

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

    const std::string msg = ::holovibes::api::get_credits();

    // Creation on the fly of the message box to display
    QMessageBox msg_box;
    msg_box.setText(QString::fromUtf8(msg.c_str()));
    msg_box.setIcon(QMessageBox::Information);
    msg_box.exec();
}

void MainWindow::documentation()
{
    LOG_INFO;
    QDesktopServices::openUrl(::holovibes::api::get_documentation_url());
}

#pragma endregion
/* ------------ */
#pragma region Ini

// FREE
void MainWindow::configure_holovibes()
{
    LOG_INFO;
    ::holovibes::api::configure_holovibes();
}

// FREE
void MainWindow::write_ini()
{
    LOG_INFO;
    ::holovibes::api::write_ini(*this);
}

// Notify
void MainWindow::write_ini(QString filename)
{
    LOG_INFO;

    ::holovibes::api::write_ini(*this, filename.toStdString());
    // Saves the current state of holovibes in holovibes.ini located in Holovibes.exe directory
    notify();
}

// Notify
void MainWindow::write_ini(const std::string& filename)
{
    LOG_INFO;

    ::holovibes::api::write_ini(*this, filename);
    // Saves the current state of holovibes in holovibes.ini located in Holovibes.exe directory
    notify();
}

// GUI
void MainWindow::browse_export_ini()
{
    LOG_INFO;

    QString filename = QFileDialog::getSaveFileName(this, tr("Save File"), "", tr("All files (*.ini)"));
    ::holovibes::api::browse_export_ini(*this, filename.toStdString());
}

// GUI
void MainWindow::browse_import_ini()
{
    LOG_INFO;
    QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("import .ini file"),
                                                    UserInterfaceDescriptor::instance().file_input_directory_.c_str(),
                                                    tr("All files (*.ini);; Ini files (*.ini)"));

    ::holovibes::api::browse_import_ini(*this, filename.toStdString());

    notify();
}

// FREE
void MainWindow::reload_ini()
{
    LOG_INFO;

    ::holovibes::api::reload_ini(*this);
}

// Notify
void MainWindow::reload_ini(QString filename)
{
    LOG_INFO;

    ::holovibes::api::reload_ini(*this, filename.toStdString());

    notify();
}

// GUI
void MainWindow::load_ini(const std::string& path)
{
    LOG_INFO;

    boost::property_tree::ptree ptree;
    boost::property_tree::ini_parser::read_ini(path, ptree);

    ::holovibes::api::load_ini(*this, path, ptree);

    GroupBox* image_rendering_group_box = ui.ImageRenderingGroupBox;
    GroupBox* view_group_box = ui.ViewGroupBox;
    GroupBox* import_group_box = ui.ImportGroupBox;
    GroupBox* info_group_box = ui.InfoGroupBox;

    QAction* image_rendering_action = ui.actionImage_rendering;
    QAction* view_action = ui.actionView;
    QAction* import_export_action = ui.actionImportExport;
    QAction* info_action = ui.actionInfo;

    if (!ptree.empty())
    {
        image_rendering_action->setChecked(
            !ptree.get<bool>("image_rendering.hidden", image_rendering_group_box->isHidden()));

        view_action->setChecked(!ptree.get<bool>("view.hidden", view_group_box->isHidden()));
        ui.ViewModeComboBox->setCurrentIndex(static_cast<int>(Holovibes::instance().get_cd().img_type.load()));
        import_export_action->setChecked(!ptree.get<bool>("import_export.hidden", import_group_box->isHidden()));

        ui.ImportInputFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));

        info_action->setChecked(!ptree.get<bool>("info.hidden", info_group_box->isHidden()));
        theme_index_ = ptree.get<int>("info.theme_type", theme_index_);
        notify();
    }
}

// GUI
void MainWindow::save_ini(const std::string& path)
{
    LOG_INFO;
    boost::property_tree::ptree ptree;

    GroupBox* image_rendering_group_box = ui.ImageRenderingGroupBox;
    GroupBox* view_group_box = ui.ViewGroupBox;
    Frame* import_export_frame = ui.ImportExportFrame;
    GroupBox* info_group_box = ui.InfoGroupBox;

    // We save in the ptree the .ini parameters that are directly linked with MainWindow ...
    ptree.put<bool>("image_rendering.hidden", image_rendering_group_box->isHidden());
    ptree.put<bool>("view.hidden", view_group_box->isHidden());

    ptree.put<bool>("import_export.hidden", import_export_frame->isHidden());

    ptree.put<bool>("info.hidden", info_group_box->isHidden());
    ptree.put<ushort>("info.theme_type", theme_index_);

    // ... then the general data to save in ptree
    ::holovibes::api::save_ini(path, ptree);

    LOG_INFO << "Configuration file holovibes.ini overwritten at " << path << std::endl;
}

#pragma endregion
/* ------------ */
#pragma region Close Compute

// GUI
void MainWindow::camera_none()
{
    LOG_INFO;

    ::holovibes::api::camera_none();

    // Make camera's settings menu unaccessible
    ui.actionSettings->setEnabled(false);

    notify();
}

// Notify
void MainWindow::reset()
{
    LOG_INFO;

    ::holovibes::api::reset(*this);

    notify();
}

// FREE
void MainWindow::closeEvent(QCloseEvent*)
{
    LOG_INFO;

    ::holovibes::api::closeEvent();
    camera_none();
    save_ini(::holovibes::ini::get_global_ini_path());
}
#pragma endregion
/* ------------ */
#pragma region Cameras

// GUI
void MainWindow::change_camera(CameraKind c)
{
    LOG_INFO;

    // Weird call to setup none camera before changing
    camera_none();

    const Computation computation = static_cast<Computation>(ui.ImageModeComboBox->currentIndex());

    const bool res = ::holovibes::api::change_camera(c, computation);

    if (res)
    {
        set_image_mode(computation);

        // Make camera's settings menu accessible
        QAction* settings = ui.actionSettings;
        settings->setEnabled(true);

        notify();
    }
}

// FREE
void MainWindow::camera_ids()
{
    LOG_INFO;
    change_camera(CameraKind::IDS);
}

// FREE
void MainWindow::camera_phantom()
{
    LOG_INFO;
    change_camera(CameraKind::Phantom);
}

// FREE
void MainWindow::camera_bitflow_cyton()
{
    LOG_INFO;
    change_camera(CameraKind::BitflowCyton);
}

// FREE
void MainWindow::camera_hamamatsu()
{
    LOG_INFO;
    change_camera(CameraKind::Hamamatsu);
}

// FREE
void MainWindow::camera_adimec()
{
    LOG_INFO;
    change_camera(CameraKind::Adimec);
}

// FREE
void MainWindow::camera_xiq()
{
    LOG_INFO;
    change_camera(CameraKind::xiQ);
}

// FREE
void MainWindow::camera_xib()
{
    LOG_INFO;
    change_camera(CameraKind::xiB);
}

// FREE
void MainWindow::configure_camera()
{
    LOG_INFO;

    ::holovibes::api::configure_camera();
}
#pragma endregion
/* ------------ */
#pragma region Image Mode
// FREE
void MainWindow::init_image_mode(QPoint& position, QSize& size)
{
    LOG_INFO;

    ::holovibes::api::init_image_mode(position, size);
}

// Notify
void MainWindow::set_raw_mode()
{
    LOG_INFO;

    ::holovibes::api::close_windows();
    ::holovibes::api::close_critical_compute();

    const bool res = ::holovibes::api::set_raw_mode(*this);

    if (res)
    {
        notify();
        layout_toggled();
    }
}

// FREE
void MainWindow::createPipe()
{
    LOG_INFO;

    ::holovibes::api::createPipe(*this);
}

// FREE
void MainWindow::createHoloWindow()
{
    LOG_INFO;

    ::holovibes::api::createHoloWindow(*this);
}

// GUI
void MainWindow::set_holographic_mode()
{
    LOG_INFO;

    // That function is used to reallocate the buffers since the Square
    // input mode could have changed
    /* Close windows & destory thread compute */
    ::holovibes::api::close_windows();
    ::holovibes::api::close_critical_compute();

    FrameDescriptor fd;
    const bool res = ::holovibes::api::set_holographic_mode(*this, fd);

    if (res)
    {
        /* Filter2D */
        ui.Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));

        /* Record Frame Calculation */
        ui.NumberOfFramesSpinBox->setValue(
            ceil((ui.ImportEndIndexSpinBox->value() - ui.ImportStartIndexSpinBox->value()) /
                 (float)ui.TimeTransformationStrideSpinBox->value()));

        /* Notify */
        notify();
    }
}

// Notify
void MainWindow::refreshViewMode()
{
    LOG_INFO;

    ::holovibes::api::refreshViewMode(*this, ui.ViewModeComboBox->currentIndex());

    notify();
    layout_toggled();
}

// LOCAL
// Is there a change in window pixel depth (needs to be re-opened)
bool MainWindow::need_refresh(const std::string& last_type, const std::string& new_type)
{
    std::vector<std::string> types_needing_refresh({"Composite image"});
    for (auto& type : types_needing_refresh)
        if ((last_type == type) != (new_type == type))
            return true;
    return false;
}

// GUI
void MainWindow::set_composite_values()
{
    const unsigned min_val_composite = Holovibes::instance().get_cd().time_transformation_size == 1 ? 0 : 1;
    const unsigned max_val_composite = Holovibes::instance().get_cd().time_transformation_size - 1;

    ui.PRedSpinBox_Composite->setValue(min_val_composite);
    ui.SpinBox_hue_freq_min->setValue(min_val_composite);
    ui.SpinBox_saturation_freq_min->setValue(min_val_composite);
    ui.SpinBox_value_freq_min->setValue(min_val_composite);

    ui.PBlueSpinBox_Composite->setValue(max_val_composite);
    ui.SpinBox_hue_freq_max->setValue(max_val_composite);
    ui.SpinBox_saturation_freq_max->setValue(max_val_composite);
    ui.SpinBox_value_freq_max->setValue(max_val_composite);
}

// GUI
std::function<void()> MainWindow::get_view_mode_callback()
{
    auto callback = ([=]() {
        Holovibes::instance().get_cd().img_type = static_cast<ImgType>(ui.ViewModeComboBox->currentIndex());
        notify();
        layout_toggled();
    });

    return callback;
}

// FREE
void MainWindow::set_view_mode(const QString value)
{
    LOG_ERROR << value.toStdString();

    if (api::is_raw_mode())
        return;

    const std::string& str = value.toStdString();

    if (need_refresh(UserInterfaceDescriptor::instance().last_img_type_, str))
    {
        refreshViewMode();
        if (Holovibes::instance().get_cd().img_type == ImgType::Composite)
        {
            set_composite_values();
        }
    }
    ::holovibes::api::set_view_mode(str, get_view_mode_callback());

    LOG_ERROR << "current image mode: " << static_cast<int>(Holovibes::instance().get_cd().img_type.load());

    // Force cuts views autocontrast if needed
    if (Holovibes::instance().get_cd().time_transformation_cuts_enabled)
        set_auto_contrast_cuts();
}

// FREE
void MainWindow::set_image_mode(QString mode)
{
    LOG_INFO;

    if (mode != nullptr)
    {
        // Call comes from ui
        if (ui.ImageModeComboBox->currentIndex() == 0)
            set_raw_mode();
        else
            set_holographic_mode();
    }
    else if (Holovibes::instance().get_cd().compute_mode == Computation::Raw)
        set_raw_mode();
    else if (Holovibes::instance().get_cd().compute_mode == Computation::Hologram)
        set_holographic_mode();
}

void MainWindow::set_image_mode(const Computation computation)
{
    if (computation == Computation::Raw)
        set_raw_mode();
    else if (computation == Computation::Hologram)
        set_holographic_mode();
}

#pragma endregion

#pragma region Batch

// GUI
void MainWindow::update_batch_size()
{
    LOG_INFO;

    uint batch_size = ui.BatchSizeSpinBox->value();
    auto callback = [=]() {
        Holovibes::instance().get_cd().batch_size = batch_size;
        ::holovibes::api::adapt_time_transformation_stride_to_batch_size();
        Holovibes::instance().get_compute_pipe()->request_update_batch_size();
        notify();
    };

    ::holovibes::api::update_batch_size(callback, batch_size);
}

#pragma endregion
/* ------------ */
#pragma region STFT

// GUI
void MainWindow::update_time_transformation_stride()
{
    LOG_INFO;

    uint time_transformation_stride = ui.TimeTransformationStrideSpinBox->value();
    auto callback = [=]() {
        Holovibes::instance().get_cd().time_transformation_stride = time_transformation_stride;
        ::holovibes::api::adapt_time_transformation_stride_to_batch_size();
        Holovibes::instance().get_compute_pipe()->request_update_time_transformation_stride();
        ui.NumberOfFramesSpinBox->setValue(
            ceil((ui.ImportEndIndexSpinBox->value() - ui.ImportStartIndexSpinBox->value()) /
                 (float)ui.TimeTransformationStrideSpinBox->value()));
        notify();
    };

    ::holovibes::api::update_time_transformation_stride(callback, time_transformation_stride);
}

// GUI
void MainWindow::toggle_time_transformation_cuts(bool checked)
{
    LOG_INFO;

    QComboBox* winSelection = ui.WindowSelectionComboBox;
    winSelection->setEnabled(checked);
    winSelection->setCurrentIndex((!checked) ? 0 : winSelection->currentIndex());

    if (!checked)
    {
        api::set_auto_contrast_cuts();
        cancel_time_transformation_cuts();
        return;
    }

    const bool res = ::holovibes::api::toggle_time_transformation_cuts(*this);

    if (res)
    {
        set_auto_contrast_cuts();
        notify();
    }
    else
    {
        cancel_time_transformation_cuts();
    }
}

// GUI
void MainWindow::cancel_time_transformation_cuts()
{
    LOG_INFO;

    std::function<void()> callback = []() { return; };

    if (auto pipe = dynamic_cast<Pipe*>(Holovibes::instance().get_compute_pipe().get()))
    {
        callback = ([=]() {
            Holovibes::instance().get_cd().time_transformation_cuts_enabled = false;
            pipe->delete_stft_slice_queue();

            ui.TimeTransformationCutsCheckBox->setChecked(false);
            notify();
        });
    }

    const bool res = ::holovibes::api::cancel_time_transformation_cuts(callback);

    if (res)
    {
        notify();
    }
}

#pragma endregion
/* ------------ */
#pragma region Computation

// Notify
void MainWindow::change_window()
{
    LOG_INFO;

    ::holovibes::api::change_window(ui.WindowSelectionComboBox->currentIndex());

    notify();
}

// FREE
void MainWindow::toggle_renormalize(bool value)
{
    LOG_INFO;

    ::holovibes::api::toggle_renormalize(value);
}

// GUI
void MainWindow::set_filter2d(bool checked)
{
    LOG_INFO;

    if (api::is_raw_mode())
        return;

    if (checked)
    {
        api::set_filter2d();

        // Set the input box related to the filter2d
        const camera::FrameDescriptor& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();
        ui.Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));
        set_filter2d_n2(ui.Filter2DN2SpinBox->value());
        set_filter2d_n1(ui.Filter2DN1SpinBox->value());
    }
    else
    {
        cancel_filter2d();
    }

    notify();
}

// GUI
void MainWindow::disable_filter2d_view()
{
    LOG_INFO;

    ::holovibes::api::disable_filter2d_view(ui.WindowSelectionComboBox->currentIndex());

    if (UserInterfaceDescriptor::instance().filter2d_window)
    {
        // Remove the on triggered event
        disconnect(UserInterfaceDescriptor::instance().filter2d_window.get(),
                   SIGNAL(destroyed()),
                   this,
                   SLOT(disable_filter2d_view()));
    }

    notify();
}

// GUI
void MainWindow::update_filter2d_view(bool checked)
{
    LOG_INFO;

    const std::optional<bool> res = ::holovibes::api::update_filter2d_view(*this, checked);

    if (res.has_value())
    {
        if (res.value())
        {
            connect(UserInterfaceDescriptor::instance().filter2d_window.get(),
                    SIGNAL(destroyed()),
                    this,
                    SLOT(disable_filter2d_view()));
        }
        notify();
    }
}

// Notify
void MainWindow::set_filter2d_n1(int n)
{
    LOG_INFO;

    const bool res = ::holovibes::api::set_filter2d_n1(n);

    if (res)
    {
        notify();
    }
}

// Notify
void MainWindow::set_filter2d_n2(int n)
{
    LOG_INFO;

    const bool res = ::holovibes::api::set_filter2d_n2(n);

    if (res)
    {
        notify();
    }
}

// Notify
void MainWindow::cancel_filter2d()
{
    LOG_INFO;

    if (api::is_raw_mode())
        return;

    api::cancel_filter2d();

    if (Holovibes::instance().get_cd().filter2d_view_enabled)
    {
        update_filter2d_view(false);
    }

    notify();
}

// FREE
void MainWindow::set_fft_shift(const bool value)
{
    LOG_INFO;

    ::holovibes::api::set_fft_shift(value);
}

// GUI
void MainWindow::set_time_transformation_size()
{
    LOG_INFO;

    int time_transformation_size = ui.timeTransformationSizeSpinBox->value();

    auto callback = [=]() {
        Holovibes::instance().get_cd().time_transformation_size = time_transformation_size;
        Holovibes::instance().get_compute_pipe()->request_update_time_transformation_size();
        set_p_accu();
        // This will not do anything until
        // SliceWindow::changeTexture() isn't coded.
    };

    const bool res = ::holovibes::api::set_time_transformation_size(time_transformation_size, callback);

    if (res)
    {
        notify();
    }
}

// GUI
void MainWindow::update_lens_view(bool value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().gpu_lens_display_enabled = value;

    if (value)
    {
        const bool res = api::set_lens_view();

        if (res)
        {
            connect(UserInterfaceDescriptor::instance().lens_window.get(),
                    SIGNAL(destroyed()),
                    this,
                    SLOT(disable_lens_view()));
        }
    }
    else
    {
        disable_lens_view();
    }
}

// GUI
void MainWindow::disable_lens_view()
{
    LOG_INFO;

    if (UserInterfaceDescriptor::instance().lens_window)
        disconnect(UserInterfaceDescriptor::instance().lens_window.get(),
                   SIGNAL(destroyed()),
                   this,
                   SLOT(disable_lens_view()));

    ::holovibes::api::disable_lens_view();

    notify();
}

// GUI
/*
void MainWindow::update_lens_view(bool value)
{
    LOG_INFO;
    Holovibes::instance().get_cd().gpu_lens_display_enabled = value;

    if (value)
    {
        const bool res = api::set_lens_view();

        if (res)
        {
            connect(UserInterfaceDescriptor::instance().lens_window.get(),
                    SIGNAL(destroyed()),
                    this,
                    SLOT(disable_lens_view()));
        }
    }
    else
    {
        disable_lens_view();
    }
}
*/
// GUI
void MainWindow::update_raw_view(bool value)
{
    LOG_INFO;

    if (value)
    {
        api::set_raw_view();
    }
    else
    {
        disable_raw_view();
    }
}

// GUI
void MainWindow::disable_raw_view()
{
    LOG_INFO;

    if (UserInterfaceDescriptor::instance().raw_window)
        disconnect(UserInterfaceDescriptor::instance().raw_window.get(),
                   SIGNAL(destroyed()),
                   this,
                   SLOT(disable_raw_view()));

    ::holovibes::api::disable_raw_view();

    notify();
}

// Notify
void MainWindow::set_p_accu()
{
    LOG_INFO;

    ::holovibes::api::set_p_accu(ui.PAccuCheckBox->isChecked(), ui.PAccSpinBox->value());

    notify();
}

// Notify
void MainWindow::set_x_accu()
{
    LOG_INFO;

    ::holovibes::api::set_x_accu(ui.XAccuCheckBox->isChecked(), ui.XAccSpinBox->value());

    notify();
}

// Notify
void MainWindow::set_y_accu()
{
    LOG_INFO;

    ::holovibes::api::set_y_accu(ui.YAccuCheckBox->isChecked(), ui.YAccSpinBox->value());

    notify();
}

// FREE
void MainWindow::set_x_y()
{
    LOG_INFO;

    const auto& fd = Holovibes::instance().get_gpu_input_queue()->get_fd();

    ::holovibes::api::set_x_y(fd, ui.XSpinBox->value(), ui.YSpinBox->value());
}

// Notify
void MainWindow::set_q(int value)
{
    LOG_INFO;

    ::holovibes::api::set_q(value);

    notify();
}

// Notify
void MainWindow::set_q_acc()
{
    LOG_INFO;

    ::holovibes::api::set_q_accu(ui.Q_AccuCheckBox->isChecked(), ui.Q_AccSpinBox->value());

    notify();
}

// Notify
void MainWindow::set_p(int value)
{
    LOG_INFO;

    const bool set_p_succeed = ::holovibes::api::set_p(value);

    if (set_p_succeed)
    {
        notify();
    }
}

// GUI
void MainWindow::set_composite_intervals()
{
    LOG_INFO;
    // PRedSpinBox_Composite value cannont be higher than PBlueSpinBox_Composite
    ui.PRedSpinBox_Composite->setValue(std::min(ui.PRedSpinBox_Composite->value(), ui.PBlueSpinBox_Composite->value()));

    ::holovibes::api::set_composite_intervals(ui.PRedSpinBox_Composite->value(), ui.PBlueSpinBox_Composite->value());

    notify();
}

// Notify
void MainWindow::set_composite_intervals_hsv_h_min()
{
    LOG_INFO;

    ::holovibes::api::set_composite_intervals_hsv_h_min(ui.SpinBox_hue_freq_min->value());

    notify();
}

// Notify
void MainWindow::set_composite_intervals_hsv_h_max()
{
    LOG_INFO;

    ::holovibes::api::set_composite_intervals_hsv_h_max(ui.SpinBox_hue_freq_max->value());

    notify();
}

// Notify
void MainWindow::set_composite_intervals_hsv_s_min()
{
    LOG_INFO;

    ::holovibes::api::set_composite_intervals_hsv_s_min(ui.SpinBox_saturation_freq_min->value());

    notify();
}

// Notify
void MainWindow::set_composite_intervals_hsv_s_max()
{
    LOG_INFO;

    ::holovibes::api::set_composite_intervals_hsv_s_max(ui.SpinBox_saturation_freq_max->value());

    notify();
}

// Notify
void MainWindow::set_composite_intervals_hsv_v_min()
{
    LOG_INFO;

    ::holovibes::api::set_composite_intervals_hsv_v_min(ui.SpinBox_value_freq_min->value());

    notify();
}

// Notify
void MainWindow::set_composite_intervals_hsv_v_max()
{
    LOG_INFO;

    ::holovibes::api::set_composite_intervals_hsv_v_max(ui.SpinBox_value_freq_max->value());

    notify();
}

// Notify
void MainWindow::set_composite_weights()
{
    LOG_INFO;

    ::holovibes::api::set_composite_weights(ui.WeightSpinBox_R->value(),
                                            ui.WeightSpinBox_G->value(),
                                            ui.WeightSpinBox_B->value());

    notify();
}

// FREE
void MainWindow::set_composite_auto_weights(bool value)
{
    LOG_INFO;

    ::holovibes::api::set_composite_auto_weights(value);
    set_auto_contrast();
}

// GUI
void MainWindow::click_composite_rgb_or_hsv()
{
    LOG_INFO;

    if (ui.radioButton_rgb->isChecked())
    {
        ::holovibes::api::select_composite_rgb();
        ui.PRedSpinBox_Composite->setValue(ui.SpinBox_hue_freq_min->value());
        ui.PBlueSpinBox_Composite->setValue(ui.SpinBox_hue_freq_max->value());
    }
    else
    {
        ::holovibes::api::select_composite_hsv();
        ui.SpinBox_hue_freq_min->setValue(ui.PRedSpinBox_Composite->value());
        ui.SpinBox_hue_freq_max->setValue(ui.PBlueSpinBox_Composite->value());
        ui.SpinBox_saturation_freq_min->setValue(ui.PRedSpinBox_Composite->value());
        ui.SpinBox_saturation_freq_max->setValue(ui.PBlueSpinBox_Composite->value());
        ui.SpinBox_value_freq_min->setValue(ui.PRedSpinBox_Composite->value());
        ui.SpinBox_value_freq_max->setValue(ui.PBlueSpinBox_Composite->value());
    }

    notify();
}

// GUI
void MainWindow::actualize_frequency_channel_s()
{
    LOG_INFO;

    ::holovibes::api::actualize_frequency_channel_s(ui.checkBox_saturation_freq->isChecked());

    ui.SpinBox_saturation_freq_min->setDisabled(!ui.checkBox_saturation_freq->isChecked());
    ui.SpinBox_saturation_freq_max->setDisabled(!ui.checkBox_saturation_freq->isChecked());
}

// GUI
void MainWindow::actualize_frequency_channel_v()
{
    LOG_INFO;

    ::holovibes::api::actualize_frequency_channel_v(ui.checkBox_value_freq->isChecked());

    ui.SpinBox_value_freq_min->setDisabled(!ui.checkBox_value_freq->isChecked());
    ui.SpinBox_value_freq_max->setDisabled(!ui.checkBox_value_freq->isChecked());
}

// GUI
void MainWindow::actualize_checkbox_h_gaussian_blur()
{
    LOG_INFO;

    ::holovibes::api::actualize_selection_h_gaussian_blur(ui.checkBox_h_gaussian_blur->isChecked());

    ui.SpinBox_hue_blur_kernel_size->setEnabled(ui.checkBox_h_gaussian_blur->isChecked());
}

// FREE
void MainWindow::actualize_kernel_size_blur()
{
    LOG_INFO;

    ::holovibes::api::actualize_kernel_size_blur(ui.SpinBox_hue_blur_kernel_size->value());
}

// LOCAL
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

// LOCAL
void slide_update_threshold(const QSlider& slider,
                            std::atomic<float>& receiver,
                            std::atomic<float>& bound_to_update,
                            QSlider& slider_to_update,
                            QLabel& to_be_written_in,
                            const std::atomic<float>& lower_bound,
                            const std::atomic<float>& upper_bound)
{

    LOG_INFO;

    const bool res =
        ::holovibes::api::slide_update_threshold(slider.value(), receiver, bound_to_update, lower_bound, upper_bound);

    char array[10];
    sprintf_s(array, "%d", slider.value());
    fancy_Qslide_text_percent(array);
    to_be_written_in.setText(QString(array));

    if (res)
    {
        slider_to_update.setValue(slider.value());
    }
}

// FREE
void MainWindow::slide_update_threshold_h_min()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_hue_threshold_min,
                           Holovibes::instance().get_cd().slider_h_threshold_min,
                           Holovibes::instance().get_cd().slider_h_threshold_max,
                           *ui.horizontalSlider_hue_threshold_max,
                           *ui.label_hue_threshold_min,
                           Holovibes::instance().get_cd().slider_h_threshold_min,
                           Holovibes::instance().get_cd().slider_h_threshold_max);
}

// FREE
void MainWindow::slide_update_threshold_h_max()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_hue_threshold_max,
                           Holovibes::instance().get_cd().slider_h_threshold_max,
                           Holovibes::instance().get_cd().slider_h_threshold_min,
                           *ui.horizontalSlider_hue_threshold_min,
                           *ui.label_hue_threshold_max,
                           Holovibes::instance().get_cd().slider_h_threshold_min,
                           Holovibes::instance().get_cd().slider_h_threshold_max);
}

// FREE
void MainWindow::slide_update_threshold_s_min()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_saturation_threshold_min,
                           Holovibes::instance().get_cd().slider_s_threshold_min,
                           Holovibes::instance().get_cd().slider_s_threshold_max,
                           *ui.horizontalSlider_saturation_threshold_max,
                           *ui.label_saturation_threshold_min,
                           Holovibes::instance().get_cd().slider_s_threshold_min,
                           Holovibes::instance().get_cd().slider_s_threshold_max);
}

// FREE
void MainWindow::slide_update_threshold_s_max()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_saturation_threshold_max,
                           Holovibes::instance().get_cd().slider_s_threshold_max,
                           Holovibes::instance().get_cd().slider_s_threshold_min,
                           *ui.horizontalSlider_saturation_threshold_min,
                           *ui.label_saturation_threshold_max,
                           Holovibes::instance().get_cd().slider_s_threshold_min,
                           Holovibes::instance().get_cd().slider_s_threshold_max);
}

// FREE
void MainWindow::slide_update_threshold_v_min()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_value_threshold_min,
                           Holovibes::instance().get_cd().slider_v_threshold_min,
                           Holovibes::instance().get_cd().slider_v_threshold_max,
                           *ui.horizontalSlider_value_threshold_max,
                           *ui.label_value_threshold_min,
                           Holovibes::instance().get_cd().slider_v_threshold_min,
                           Holovibes::instance().get_cd().slider_v_threshold_max);
}

// FREE
void MainWindow::slide_update_threshold_v_max()
{
    LOG_INFO;
    slide_update_threshold(*ui.horizontalSlider_value_threshold_max,
                           Holovibes::instance().get_cd().slider_v_threshold_max,
                           Holovibes::instance().get_cd().slider_v_threshold_min,
                           *ui.horizontalSlider_value_threshold_min,
                           *ui.label_value_threshold_max,
                           Holovibes::instance().get_cd().slider_v_threshold_min,
                           Holovibes::instance().get_cd().slider_v_threshold_max);
}

// Notify
void MainWindow::increment_p()
{
    LOG_INFO;

    bool res = ::holovibes::api::increment_p();

    if (res)
    {
        set_auto_contrast();
        notify();
    }
}

// Notify
void MainWindow::decrement_p()
{
    LOG_INFO;

    bool res = ::holovibes::api::decrement_p();

    if (res)
    {
        set_auto_contrast();
        notify();
    }
}

// FREE
void MainWindow::set_wavelength(const double value)
{
    LOG_INFO;

    ::holovibes::api::set_wavelength(value);
}

// FREE
void MainWindow::set_z(const double value)
{
    LOG_INFO;

    ::holovibes::api::set_z(value);
}

// GUI
void MainWindow::increment_z()
{
    LOG_INFO;

    const bool res = ::holovibes::api::is_raw_mode();

    if (res)
    {
        ui.ZDoubleSpinBox->setValue(Holovibes::instance().get_cd().zdistance);
    }
}

// GUI
void MainWindow::decrement_z()
{
    LOG_INFO;

    bool res = ::holovibes::api::decrement_z();

    if (res)
    {
        ui.ZDoubleSpinBox->setValue(Holovibes::instance().get_cd().zdistance);
    }
}

// GUI
void MainWindow::set_z_step(const double value)
{
    LOG_INFO;

    ::holovibes::api::set_z_step(value);

    ui.ZDoubleSpinBox->setSingleStep(value);
}

// FREE
void MainWindow::set_space_transformation(const QString value)
{
    LOG_INFO;

    const bool res = ::holovibes::api::set_space_transformation(value.toStdString());

    if (res)
    {
        set_holographic_mode();
    }
}

// FREE
void MainWindow::set_time_transformation(QString value)
{
    LOG_INFO;

    const bool res = ::holovibes::api::set_time_transformation(value.toStdString());

    if (res)
    {
        set_holographic_mode();
    }
}

// Notify
void MainWindow::set_unwrapping_2d(const bool value)
{
    LOG_INFO;

    bool res = ::holovibes::api::set_unwrapping_2d(value);

    if (res)
    {
        notify();
    }
}

// Notify
void MainWindow::set_accumulation(bool value)
{
    LOG_INFO;
    bool res = ::holovibes::api::set_accumulation(value);

    if (res)
    {
        notify();
    }
}

// FREE
void MainWindow::set_accumulation_level(int value)
{
    LOG_INFO;

    ::holovibes::api::set_accumulation_level(value);
}

// FREE
void MainWindow::set_composite_area()
{
    LOG_INFO;

    ::holovibes::api::set_composite_area();
}

#pragma endregion
/* ------------ */
#pragma region Texture

// Notify
void MainWindow::rotateTexture()
{
    LOG_INFO;

    ::holovibes::api::rotateTexture();

    notify();
}

// Notify
void MainWindow::flipTexture()
{
    LOG_INFO;

    ::holovibes::api::flipTexture();

    notify();
}

#pragma endregion
/* ------------ */
#pragma region Contrast - Log

// Notify
void MainWindow::set_contrast_mode(bool value)
{
    LOG_INFO;

    change_window();

    bool res = ::holovibes::api::set_contrast_mode(value);

    if (res)
    {
        notify();
    }
}

// FREE
void MainWindow::set_auto_contrast_cuts()
{
    LOG_INFO;

    ::holovibes::api::set_auto_contrast_cuts();
}

// GUI
void MainWindow::QSpinBoxQuietSetValue(QSpinBox* spinBox, int value)
{
    LOG_INFO;
    spinBox->blockSignals(true);
    spinBox->setValue(value);
    spinBox->blockSignals(false);
}

// GUI
void MainWindow::QSliderQuietSetValue(QSlider* slider, int value)
{
    LOG_INFO;
    slider->blockSignals(true);
    slider->setValue(value);
    slider->blockSignals(false);
}

// GUI
void MainWindow::QDoubleSpinBoxQuietSetValue(QDoubleSpinBox* spinBox, double value)
{
    LOG_INFO;
    spinBox->blockSignals(true);
    spinBox->setValue(value);
    spinBox->blockSignals(false);
}

// FREE
void MainWindow::set_auto_contrast()
{
    LOG_INFO;

    ::holovibes::api::set_auto_contrast();
}

// FREE
void MainWindow::set_contrast_min(const double value)
{
    LOG_INFO;

    ::holovibes::api::set_contrast_min(value);
}

// FREE
void MainWindow::set_contrast_max(const double value)
{
    LOG_INFO;

    ::holovibes::api::set_contrast_max(value);
}

// FREE
void MainWindow::invert_contrast(bool value)
{
    LOG_INFO;

    ::holovibes::api::invert_contrast(value);
}

// Notify
void MainWindow::set_auto_refresh_contrast(bool value)
{
    LOG_INFO;

    ::holovibes::api::set_auto_refresh_contrast(value);

    notify();
}

// Notify
void MainWindow::set_log_scale(const bool value)
{
    LOG_INFO;

    const bool res = ::holovibes::api::set_log_scale(value);

    if (res)
    {
        notify();
    }
}
#pragma endregion
/* ------------ */
#pragma region Convolution

// Notify
void MainWindow::update_convo_kernel(const QString& value)
{
    LOG_INFO;

    bool res = ::holovibes::api::update_convo_kernel(value.toStdString());

    if (res)
    {
        notify();
    }
}

// Notify
void MainWindow::set_convolution_mode(const bool value)
{
    LOG_INFO;

    if (value)
    {
        std::string str = ui.KernelQuickSelectComboBox->currentText().toStdString();

        ::holovibes::api::set_convolution_mode(str);
    }
    else
    {
        ::holovibes::api::unset_convolution_mode();
    }

    notify();
}

// Notify
void MainWindow::set_divide_convolution_mode(const bool value)
{
    LOG_INFO;

    ::holovibes::api::set_divide_convolution_mode(value);

    notify();
}

#pragma endregion
/* ------------ */
#pragma region Reticle

// Notify
void MainWindow::display_reticle(bool value)
{
    LOG_INFO;

    ::holovibes::api::display_reticle(value);

    notify();
}

// FREE
void MainWindow::reticle_scale(double value)
{
    LOG_INFO;

    ::holovibes::api::reticle_scale(value);
}
#pragma endregion Reticle
/* ------------ */
#pragma region Chart

// Notify
void MainWindow::activeSignalZone()
{
    LOG_INFO;

    ::holovibes::api::activeSignalZone();

    notify();
}

// Notify
void MainWindow::activeNoiseZone()
{
    LOG_INFO;

    ::holovibes::api::activeNoiseZone();

    notify();
}

// GUI
void MainWindow::start_chart_display()
{
    LOG_INFO;

    holovibes::api::start_chart_display();

    connect(UserInterfaceDescriptor::instance().plot_window_.get(),
            SIGNAL(closed()),
            this,
            SLOT(stop_chart_display()),
            Qt::UniqueConnection);
    ui.ChartPlotPushButton->setEnabled(false);
}

// GUI
void MainWindow::stop_chart_display()
{
    LOG_INFO;

    holovibes::api::stop_chart_display();

    ui.ChartPlotPushButton->setEnabled(true);
}
#pragma endregion
/* ------------ */
#pragma region Record

// GUI
void MainWindow::set_record_frame_step(int value)
{
    LOG_INFO;

    ::holovibes::api::set_record_frame_step(value);

    ui.NumberOfFramesSpinBox->setSingleStep(value);
}

// GUI
void MainWindow::set_nb_frames_mode(bool value)
{
    LOG_INFO;

    ui.NumberOfFramesSpinBox->setEnabled(value);
}

// GUI
void MainWindow::browse_record_output_file()
{
    LOG_INFO;
    QString filepath;

    // Open file explorer dialog on the fly depending on the record mode
    // Add the matched extension to the file if none
    if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Chart output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Text files (*.txt);;CSV files (*.csv)"));
    }
    else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::RAW)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo)"));
    }
    else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::HOLOGRAM)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                UserInterfaceDescriptor::instance().record_output_directory_.c_str(),
                                                tr("Holo files (*.holo);; Avi Files (*.avi);; Mp4 files (*.mp4)"));
    }

    if (filepath.isEmpty())
        return;

    // Convert QString to std::string
    std::string std_filepath = filepath.toStdString();

    const std::string file_ext = ::holovibes::api::browse_record_output_file(std_filepath);

    // Will pick the item combobox related to file_ext if it exists, else, nothing is done
    ui.RecordExtComboBox->setCurrentText(file_ext.c_str());

    notify();
}

// GUI
void MainWindow::browse_batch_input()
{
    LOG_INFO;

    // Open file explorer on the fly
    QString filename = QFileDialog::getOpenFileName(this,
                                                    tr("Batch input file"),
                                                    UserInterfaceDescriptor::instance().batch_input_directory_.c_str(),
                                                    tr("All files (*)"));

    // Output the file selected in he ui line edit widget
    QLineEdit* batch_input_line_edit = ui.BatchInputPathLineEdit;
    batch_input_line_edit->clear();
    batch_input_line_edit->insert(filename);
}

// GUI
void MainWindow::set_record_mode(const QString& value)
{
    LOG_INFO;
    if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
        stop_chart_display();

    ::holovibes::api::stop_record();

    const std::string text = value.toStdString();
    ::holovibes::api::set_record_mode(text);

    if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::CHART)
    {
        ui.RecordExtComboBox->clear();
        ui.RecordExtComboBox->insertItem(0, ".csv");
        ui.RecordExtComboBox->insertItem(1, ".txt");

        ui.ChartPlotWidget->show();

        if (UserInterfaceDescriptor::instance().mainDisplay)
        {
            UserInterfaceDescriptor::instance().mainDisplay->resetTransform();

            UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().enable_all(Signal);
            UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().enable_all(Noise);
            UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().create_overlay<Signal>();
        }
    }
    else
    {
        if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::RAW)
        {
            ui.RecordExtComboBox->clear();
            ui.RecordExtComboBox->insertItem(0, ".holo");
        }
        else if (UserInterfaceDescriptor::instance().record_mode_ == RecordMode::HOLOGRAM)
        {
            ui.RecordExtComboBox->clear();
            ui.RecordExtComboBox->insertItem(0, ".holo");
            ui.RecordExtComboBox->insertItem(1, ".avi");
            ui.RecordExtComboBox->insertItem(2, ".mp4");
        }

        ui.ChartPlotWidget->hide();

        if (UserInterfaceDescriptor::instance().mainDisplay)
        {
            UserInterfaceDescriptor::instance().mainDisplay->resetTransform();

            UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(Signal);
            UserInterfaceDescriptor::instance().mainDisplay->getOverlayManager().disable_all(Noise);
        }
    }

    notify();
}

// FREE
void MainWindow::stop_record()
{
    LOG_INFO;
    ::holovibes::api::stop_record();
}

// GUI
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
    ui.BatchSizeSpinBox->setEnabled(Holovibes::instance().get_cd().compute_mode == Computation::Hologram);
    ::holovibes::api::record_finished();
}

// GUI
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

    std::string batch_input_path = ui.BatchInputPathLineEdit->text().toStdString();

    // Preconditions to start record
    const bool preconditions = ::holovibes::api::start_record_preconditions(batch_enabled,
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
    UserInterfaceDescriptor::instance().raw_window.reset(nullptr);
    disable_raw_view();
    ui.RawDisplayingCheckBox->setHidden(true);

    ui.BatchSizeSpinBox->setEnabled(false);
    UserInterfaceDescriptor::instance().is_recording_ = true;

    ui.ExportRecPushButton->setEnabled(false);
    ui.ExportStopPushButton->setEnabled(true);

    ui.RecordProgressBar->reset();
    ui.RecordProgressBar->show();

    auto callback = [record_mode = UserInterfaceDescriptor::instance().record_mode_, this]() {
        synchronize_thread([=]() { record_finished(record_mode); });
    };

    ::holovibes::api::start_record(batch_enabled, nb_frames_to_record, output_path, batch_input_path, callback);
}
#pragma endregion
/* ------------ */
#pragma region Import

// GUI
void MainWindow::set_start_stop_buttons(bool value)
{
    LOG_INFO;
    ui.ImportStartPushButton->setEnabled(value);
    ui.ImportStopPushButton->setEnabled(value);
}

// GUI
void MainWindow::import_browse_file()
{
    LOG_INFO;
    QString filename = "";
    // Open the file explorer to let the user pick his file
    // and store the chosen file in filename

    filename = QFileDialog::getOpenFileName(this,
                                            tr("import file"),
                                            UserInterfaceDescriptor::instance().file_input_directory_.c_str(),
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
        UserInterfaceDescriptor::instance().file_fd_ = input_file->get_frame_descriptor();
        input_file->import_compute_settings(Holovibes::instance().get_cd());

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

// TODO: method should be a simple call to API
// GUI
void MainWindow::import_stop()
{
    LOG_INFO;
    ::holovibes::api::close_windows();
    cancel_time_transformation_cuts();

    ::holovibes::api::import_stop();
    // FIXME: import_stop() and camera_none() call same methods
    // FIXME: camera_none() weird call because we are dealing with imported file
    camera_none();
    synchronize_thread([&]() { ui.FileReaderProgressBar->hide(); });
    notify();
}

// GUI
void MainWindow::import_start()
{
    LOG_INFO;

    // Check if computation is currently running
    if (!Holovibes::instance().get_cd().is_computation_stopped)
    {
        import_stop();
    }

    // shift main window when camera view appears
    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();
    move(QPoint(210 + (screen_width - 800) / 2, 200 + (screen_height - 500) / 2));

    QLineEdit* import_line_edit = ui.ImportPathLineEdit;
    QSpinBox* fps_spinbox = ui.ImportInputFpsSpinBox;
    start_spinbox = ui.ImportStartIndexSpinBox;
    QCheckBox* load_file_gpu_box = ui.LoadFileInGpuCheckBox;
    end_spinbox = ui.ImportEndIndexSpinBox;

    std::string file_path = import_line_edit->text().toStdString();

    bool res_import_start = ::holovibes::api::import_start(file_path,
                                                           fps_spinbox->value(),
                                                           start_spinbox->value(),
                                                           load_file_gpu_box->isChecked(),
                                                           end_spinbox->value());

    if (res_import_start)
    {
        ui.FileReaderProgressBar->show();
        UserInterfaceDescriptor::instance().is_enabled_camera_ = true;
        set_image_mode(nullptr);

        // Make camera's settings menu unaccessible
        QAction* settings = ui.actionSettings;
        settings->setEnabled(false);

        UserInterfaceDescriptor::instance().import_type_ = ::holovibes::UserInterfaceDescriptor::ImportType::File;

        notify();
    }
    else
    {
        UserInterfaceDescriptor::instance().mainDisplay.reset(nullptr);
    }

    ui.ImageModeComboBox->setCurrentIndex(::holovibes::api::is_raw_mode() ? 0 : 1);
}

// GUI
void MainWindow::import_start_spinbox_update()
{
    LOG_INFO;
    QSpinBox* start_spinbox = ui.ImportStartIndexSpinBox;
    QSpinBox* end_spinbox = ui.ImportEndIndexSpinBox;

    if (start_spinbox->value() > end_spinbox->value())
        end_spinbox->setValue(start_spinbox->value());
}

// GUI
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

// GUI
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
    theme_index_ = 1;
}

// GUI
void MainWindow::set_classic()
{
    LOG_INFO;
    qApp->setPalette(this->style()->standardPalette());
    // Light mode style
    qApp->setStyle(QStyleFactory::create("WindowsVista"));
    qApp->setStyleSheet("");
    theme_index_ = 0;
}
#pragma endregion

#pragma region Getters

// LOCAL
RawWindow* MainWindow::get_main_display()
{
    LOG_INFO;
    return UserInterfaceDescriptor::instance().mainDisplay.get();
}

// GUI
void MainWindow::update_file_reader_index(int n)
{
    LOG_INFO;
    auto lambda = [this, n]() { ui.FileReaderProgressBar->setValue(n); };
    synchronize_thread(lambda);
}
#pragma endregion
} // namespace gui
} // namespace holovibes
