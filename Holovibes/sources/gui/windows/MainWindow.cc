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
    catch (std::exception&)
    {
        LOG_WARN << ::holovibes::ini::get_global_ini_path() << ": Configuration file not found. "
                 << "Initialization with default values.";
        save_ini(::holovibes::ini::get_global_ini_path());
    }

    set_z_step(z_step_);
    set_record_frame_step(record_frame_step_);
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
    cd_.compute_mode = Computation::Raw;
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
        ui.CompositeGroupBox->hide();
        ui.ImageRenderingGroupBox->setEnabled(false);
        ui.ViewGroupBox->setEnabled(false);
        ui.ExportGroupBox->setEnabled(false);
        layout_toggled();
        return;
    }

    if (cd_.compute_mode == Computation::Raw && is_enabled_camera_)
    {
        ui.ImageRenderingGroupBox->setEnabled(true);
        ui.ViewGroupBox->setEnabled(false);
        ui.ExportGroupBox->setEnabled(true);
    }

    else if (cd_.compute_mode == Computation::Hologram && is_enabled_camera_)
    {
        ui.ImageRenderingGroupBox->setEnabled(true);
        ui.ViewGroupBox->setEnabled(true);
        ui.ExportGroupBox->setEnabled(true);
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
        (signalBtn->isEnabled() && mainDisplay && mainDisplay->getKindOfOverlay() == KindOfOverlay::Signal)
            ? "QPushButton {color: #8E66D9;}"
            : "");

    QPushButton* noiseBtn = ui.ChartNoisePushButton;
    noiseBtn->setStyleSheet(
        (noiseBtn->isEnabled() && mainDisplay && mainDisplay->getKindOfOverlay() == KindOfOverlay::Noise)
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
    ui.ContrastMinDoubleSpinBox->setValue(cd_.get_contrast_min(cd_.current_window));
    ui.ContrastMaxDoubleSpinBox->setEnabled(!cd_.contrast_auto_refresh);
    ui.ContrastMaxDoubleSpinBox->setValue(cd_.get_contrast_max(cd_.current_window));

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
    if (cd_.p_acc_level > cd_.time_transformation_size - 1)
        cd_.p_acc_level = cd_.time_transformation_size - 1;
    ui.PAccSpinBox->setValue(cd_.p_acc_level);
    ui.PAccSpinBox->setEnabled(cd_.img_type != ImgType::PhaseIncrease);
    if (cd_.p_accu_enabled)
    {
        ui.PSpinBox->setMaximum(cd_.time_transformation_size - cd_.p_acc_level - 1);
        if (cd_.pindex > cd_.time_transformation_size - cd_.p_acc_level - 1)
            cd_.pindex = cd_.time_transformation_size - cd_.p_acc_level - 1;
        ui.PSpinBox->setValue(cd_.pindex);
        ui.PAccSpinBox->setMaximum(cd_.time_transformation_size - cd_.pindex - 1);
    }
    else
    {
        ui.PSpinBox->setMaximum(cd_.time_transformation_size - 1);
        if (cd_.pindex > cd_.time_transformation_size - 1)
            cd_.pindex = cd_.time_transformation_size - 1;
        ui.PSpinBox->setValue(cd_.pindex);
    }
    ui.PSpinBox->setEnabled(!is_raw);

    // q accu
    bool is_ssa_stft = cd_.time_transformation == TimeTransformation::SSA_STFT;
    ui.Q_AccuCheckBox->setEnabled(!is_raw && is_ssa_stft);
    ui.Q_AccSpinBox->setEnabled(!is_raw && is_ssa_stft);
    ui.Q_SpinBox->setEnabled(!is_raw && is_ssa_stft);

    ui.Q_AccuCheckBox->setChecked(cd_.q_acc_enabled);
    ui.Q_AccSpinBox->setMaximum(cd_.time_transformation_size - 1);
    if (cd_.q_acc_level > cd_.time_transformation_size - 1)
        cd_.q_acc_level = cd_.time_transformation_size - 1;
    ui.Q_AccSpinBox->setValue(cd_.q_acc_level);
    if (cd_.q_acc_enabled)
    {
        ui.Q_SpinBox->setMaximum(cd_.time_transformation_size - cd_.q_acc_level - 1);
        if (cd_.q_index > cd_.time_transformation_size - cd_.q_acc_level - 1)
            cd_.q_index = cd_.time_transformation_size - cd_.q_acc_level - 1;
        ui.Q_SpinBox->setValue(cd_.q_index);
        ui.Q_AccSpinBox->setMaximum(cd_.time_transformation_size - cd_.q_index - 1);
    }
    else
    {
        ui.Q_SpinBox->setMaximum(cd_.time_transformation_size - 1);
        if (cd_.q_index > cd_.time_transformation_size - 1)
            cd_.q_index = cd_.time_transformation_size - 1;
        ui.Q_SpinBox->setValue(cd_.q_index);
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
    ui.TimeTransformationStrideSpinBox->setEnabled(!cd_.fast_pipe && !is_raw);

    const uint input_queue_capacity = global::global_config.input_queue_max_size;

    ui.TimeTransformationStrideSpinBox->setValue(cd_.time_transformation_stride);
    ui.TimeTransformationStrideSpinBox->setSingleStep(cd_.batch_size);
    ui.TimeTransformationStrideSpinBox->setMinimum(cd_.batch_size);

    // Batch
    ui.BatchSizeSpinBox->setEnabled(!cd_.fast_pipe && !is_raw && !is_recording_);

    if (cd_.batch_size > input_queue_capacity)
        cd_.batch_size = input_queue_capacity;

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
    ui.timeTransformationSizeSpinBox->setEnabled(!cd_.fast_pipe && !is_raw && !cd_.time_transformation_cuts_enabled);
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
    actualize_frequency_channel_v();

    QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_min, cd_.composite_p_min_h);
    QSpinBoxQuietSetValue(ui.SpinBox_hue_freq_max, cd_.composite_p_max_h);
    QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_min, (int)(cd_.slider_h_threshold_min * 1000));
    slide_update_threshold_h_min();
    QSliderQuietSetValue(ui.horizontalSlider_hue_threshold_max, (int)(cd_.slider_h_threshold_max * 1000));
    slide_update_threshold_h_max();

    QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_min, cd_.composite_p_min_s);
    QSpinBoxQuietSetValue(ui.SpinBox_saturation_freq_max, cd_.composite_p_max_s);
    QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_min, (int)(cd_.slider_s_threshold_min * 1000));
    slide_update_threshold_s_min();
    QSliderQuietSetValue(ui.horizontalSlider_saturation_threshold_max, (int)(cd_.slider_s_threshold_max * 1000));
    slide_update_threshold_s_max();

    QSpinBoxQuietSetValue(ui.SpinBox_value_freq_min, cd_.composite_p_min_v);
    QSpinBoxQuietSetValue(ui.SpinBox_value_freq_max, cd_.composite_p_max_v);
    QSliderQuietSetValue(ui.horizontalSlider_value_threshold_min, (int)(cd_.slider_v_threshold_min * 1000));
    slide_update_threshold_v_min();
    QSliderQuietSetValue(ui.horizontalSlider_value_threshold_max, (int)(cd_.slider_v_threshold_max * 1000));
    slide_update_threshold_v_max();

    ui.CompositeGroupBox->setHidden(is_raw_mode() || (cd_.img_type != ImgType::Composite));

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

void MainWindow::notify_error(std::exception& e)
{
    CustomException* err_ptr = dynamic_cast<CustomException*>(&e);
    if (err_ptr)
    {
        UpdateException* err_update_ptr = dynamic_cast<UpdateException*>(err_ptr);
        if (err_update_ptr)
        {
            auto lambda = [this] {
                // notify will be in close_critical_compute
                cd_.pindex = 0;
                cd_.time_transformation_size = 1;
                if (cd_.convolution_enabled)
                {
                    cd_.convolution_enabled = false;
                }
                close_windows();
                close_critical_compute();
                display_error("GPU computing error occured.\n");
                notify();
            };
            synchronize_thread(lambda);
        }

        auto lambda = [this, accu = (dynamic_cast<AccumulationException*>(err_ptr) != nullptr)] {
            if (accu)
            {
                cd_.img_acc_slice_xy_enabled = false;
                cd_.img_acc_slice_xy_level = 1;
            }
            close_critical_compute();

            display_error("GPU computing error occured.\n");
            notify();
        };
        synchronize_thread(lambda);
    }
    else
    {
        display_error("Unknown error occured.");
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

void MainWindow::display_error(const std::string msg) { LOG_ERROR << msg; }

void MainWindow::display_info(const std::string msg) { LOG_INFO << msg; }

void MainWindow::credits()
{
    std::string msg = "Holovibes " + version +
                      "\n\n"

                      "Developers:\n\n"

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

void MainWindow::write_ini()
{
    // Saves the current state of holovibes in holovibes.ini located in Holovibes.exe directory
    save_ini(GLOBAL_INI_PATH);
    notify();
}

void MainWindow::reload_ini()
{
    import_stop();
    try
    {
        load_ini(::holovibes::ini::get_global_ini_path());
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
    }
    if (import_type_ == ImportType::File)
    {
        import_start();
    }
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

        view_action->setChecked(!ptree.get<bool>("view.hidden", view_group_box->isHidden()));

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
        set_record_frame_step(record_frame_step);

        import_export_action->setChecked(!ptree.get<bool>("import_export.hidden", import_group_box->isHidden()));

        ui.ImportInputFpsSpinBox->setValue(ptree.get<int>("import.fps", 60));

        info_action->setChecked(!ptree.get<bool>("info.hidden", info_group_box->isHidden()));
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
    GroupBox* view_group_box = ui.ViewGroupBox;
    Frame* import_export_frame = ui.ImportExportFrame;
    GroupBox* info_group_box = ui.InfoGroupBox;
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

    ptree.put<bool>("view.hidden", view_group_box->isHidden());

    ptree.put<float>("view.mainWindow_rotate", displayAngle);
    ptree.put<float>("view.xCut_rotate", xzAngle);
    ptree.put<float>("view.yCut_rotate", yzAngle);
    ptree.put<int>("view.mainWindow_flip", displayFlip);
    ptree.put<int>("view.xCut_flip", xzFlip);
    ptree.put<int>("view.yCut_flip", yzFlip);

    ptree.put<size_t>("chart.auto_scale_point_threshold", auto_scale_point_threshold_);

    ptree.put<uint>("record.record_frame_step", record_frame_step_);

    ptree.put<bool>("import_export.hidden", import_export_frame->isHidden());

    ptree.put<bool>("info.hidden", info_group_box->isHidden());
    ptree.put<ushort>("info.theme_type", theme_index_);

    ptree.put<uint>("display.main_window_max_size", window_max_size);
    ptree.put<uint>("display.time_transformation_cuts_window_max_size", time_transformation_cuts_window_max_size);
    ptree.put<uint>("display.auxiliary_window_max_size", auxiliary_window_max_size);

    boost::property_tree::write_ini(path, ptree);

    LOG_INFO << "Configuration file holovibes.ini overwritten";
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
        cancel_time_transformation_cuts();

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

    cd_.is_computation_stopped = true;
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
    cd_.gpu_lens_display_enabled = false;

    filter2d_window.reset(nullptr);
    cd_.filter2d_view_enabled = false;

    /* Raw view & recording */
    raw_window.reset(nullptr);
    cd_.raw_view_enabled = false;

    // Disable overlays
    cd_.reticle_enabled = false;
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
    cd_.pindex = 0;
    cd_.time_transformation_size = 1;
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
    try
    {
        load_ini(::holovibes::ini::get_global_ini_path());
    }
    catch (std::exception&)
    {
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

            cd_.is_computation_stopped = false;
            notify();
        }
        catch (camera::CameraException& e)
        {
            display_error("[CAMERA] " + std::string(e.what()));
        }
        catch (std::exception& e)
        {
            display_error(e.what());
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
        cd_.compute_mode = Computation::Raw;
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
    catch (std::runtime_error& e)
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
    catch (std::runtime_error& e)
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
        cd_.compute_mode = Computation::Hologram;
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
        cd_.contrast_enabled = true;

        /* Filter2D */
        ui.Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));

        /* Record Frame Calculation */
        ui.NumberOfFramesSpinBox->setValue(
            ceil((ui.ImportEndIndexSpinBox->value() - ui.ImportStartIndexSpinBox->value()) /
                 (float)ui.TimeTransformationStrideSpinBox->value()));

        /* Notify */
        notify();
    }
    catch (std::runtime_error& e)
    {

        LOG_ERROR << "cannot set holographic mode: " << e.what();
    }
}

void MainWindow::set_computation_mode()
{
    if (ui.ImageModeComboBox->currentIndex() == 0)
    {
        cd_.compute_mode = Computation::Raw;
    }
    else if (ui.ImageModeComboBox->currentIndex() == 1)
    {
        cd_.compute_mode = Computation::Hologram;
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
    catch (std::runtime_error& e)
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
void MainWindow::set_view_mode(const QString value)
{
    if (!is_raw_mode())
    {
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
            cd_.img_type = static_cast<ImgType>(ui.ViewModeComboBox->currentIndex());
            notify();
            layout_toggled();
        });
        pipe_refresh();

        // Force XYview autocontrast
        pipe->autocontrast_end_pipe(WindowKind::XYview);
        // Force cuts views autocontrast if needed
        if (cd_.time_transformation_cuts_enabled)
            set_auto_contrast_cuts();
    }
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
    if (!is_raw_mode())
    {
        int value = ui.BatchSizeSpinBox->value();

        if (value == cd_.batch_size)
            return;

        auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());
        if (pipe)
        {
            pipe->insert_fn_end_vect([=]() {
                cd_.batch_size = value;
                adapt_time_transformation_stride_to_batch_size(cd_);
                holovibes_.get_compute_pipe()->request_update_batch_size();
                notify();
            });
        }
        else
            std::cout << "COULD NOT GET PIPE" << std::endl;
    }
}

#pragma endregion
/* ------------ */
#pragma region STFT
void MainWindow::cancel_stft_slice_view()
{
    cd_.contrast_max_slice_xz = false;
    cd_.contrast_max_slice_yz = false;
    cd_.log_scale_slice_xz_enabled = false;
    cd_.log_scale_slice_yz_enabled = false;
    cd_.img_acc_slice_xz_enabled = false;
    cd_.img_acc_slice_yz_enabled = false;
    sliceXZ.reset(nullptr);
    sliceYZ.reset(nullptr);

    if (mainDisplay)
    {
        mainDisplay->setCursor(Qt::ArrowCursor);
        mainDisplay->getOverlayManager().disable_all(SliceCross);
        mainDisplay->getOverlayManager().disable_all(Cross);
    }
    if (auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get()))
    {
        pipe->insert_fn_end_vect([=]() {
            cd_.time_transformation_cuts_enabled = false;
            pipe->delete_stft_slice_queue();

            ui.TimeTransformationCutsCheckBox->setChecked(false);
            notify();
        });
    }
}

void MainWindow::update_time_transformation_stride()
{
    if (!is_raw_mode())
    {
        int value = ui.TimeTransformationStrideSpinBox->value();

        if (value == cd_.time_transformation_stride)
            return;

        auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());
        if (pipe)
        {
            pipe->insert_fn_end_vect([=]() {
                cd_.time_transformation_stride = value;
                adapt_time_transformation_stride_to_batch_size(cd_);
                holovibes_.get_compute_pipe()->request_update_time_transformation_stride();
                ui.NumberOfFramesSpinBox->setValue(
                    ceil((ui.ImportEndIndexSpinBox->value() - ui.ImportStartIndexSpinBox->value()) /
                         (float)ui.TimeTransformationStrideSpinBox->value()));
                notify();
            });
        }
        else
            std::cout << "COULD NOT GET PIPE" << std::endl;
    }
}

void MainWindow::toggle_time_transformation_cuts(bool checked)
{
    QComboBox* winSelection = ui.WindowSelectionComboBox;
    winSelection->setEnabled(checked);
    winSelection->setCurrentIndex((!checked) ? 0 : winSelection->currentIndex());
    if (checked)
    {
        try
        {
            holovibes_.get_compute_pipe()->create_stft_slice_queue();
            // set positions of new windows according to the position of the
            // main GL window
            QPoint xzPos = mainDisplay->framePosition() + QPoint(0, mainDisplay->height() + 42);
            QPoint yzPos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 20, 0);
            const ushort nImg = cd_.time_transformation_size;
            uint time_transformation_size = std::max(256u, std::min(512u, (uint)nImg));

            if (time_transformation_size > time_transformation_cuts_window_max_size)
                time_transformation_size = time_transformation_cuts_window_max_size;

            while (holovibes_.get_compute_pipe()->get_update_time_transformation_size_request())
                continue;
            while (holovibes_.get_compute_pipe()->get_cuts_request())
                continue;
            sliceXZ.reset(new SliceWindow(xzPos,
                                          QSize(mainDisplay->width(), time_transformation_size),
                                          holovibes_.get_compute_pipe()->get_stft_slice_queue(0).get(),
                                          KindOfView::SliceXZ,
                                          this));
            sliceXZ->setTitle("XZ view");
            sliceXZ->setAngle(xzAngle);
            sliceXZ->setFlip(xzFlip);
            sliceXZ->setCd(&cd_);

            sliceYZ.reset(new SliceWindow(yzPos,
                                          QSize(time_transformation_size, mainDisplay->height()),
                                          holovibes_.get_compute_pipe()->get_stft_slice_queue(1).get(),
                                          KindOfView::SliceYZ,
                                          this));
            sliceYZ->setTitle("YZ view");
            sliceYZ->setAngle(yzAngle);
            sliceYZ->setFlip(yzFlip);
            sliceYZ->setCd(&cd_);

            mainDisplay->getOverlayManager().create_overlay<Cross>();
            cd_.time_transformation_cuts_enabled = true;
            set_auto_contrast_cuts();
            auto holo = dynamic_cast<HoloWindow*>(mainDisplay.get());
            if (holo)
                holo->update_slice_transforms();
            notify();
        }
        catch (std::logic_error& e)
        {
            std::cerr << e.what() << std::endl;
            cancel_stft_slice_view();
        }
    }
    else
    {
        cancel_stft_slice_view();
    }
}

void MainWindow::cancel_time_transformation_cuts()
{
    if (cd_.time_transformation_cuts_enabled)
    {
        cancel_stft_slice_view();
        try
        {
            // Wait for refresh to be enabled for notify
            while (holovibes_.get_compute_pipe()->get_refresh_request())
                continue;
        }
        catch (std::exception&)
        {
        }
        cd_.time_transformation_cuts_enabled = false;
    }
    notify();
}

#pragma endregion
/* ------------ */
#pragma region Computation
void MainWindow::change_window()
{
    QComboBox* window_cbox = ui.WindowSelectionComboBox;

    if (window_cbox->currentIndex() == 0)
        cd_.current_window = WindowKind::XYview;
    else if (window_cbox->currentIndex() == 1)
        cd_.current_window = WindowKind::XZview;
    else if (window_cbox->currentIndex() == 2)
        cd_.current_window = WindowKind::YZview;
    else if (window_cbox->currentIndex() == 3)
        cd_.current_window = WindowKind::Filter2D;
    pipe_refresh();
    notify();
}

void MainWindow::toggle_renormalize(bool value)
{
    cd_.renorm_enabled = value;

    holovibes_.get_compute_pipe()->request_clear_img_acc();
    pipe_refresh();
}

void MainWindow::set_filter2d(bool checked)
{
    if (!is_raw_mode())
    {
        if (checked == false)
        {
            cd_.filter2d_enabled = checked;
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
            cd_.filter2d_enabled = checked;
        }
        pipe_refresh();
        notify();
    }
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
    if (!is_raw_mode())
    {
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
            catch (std::exception& e)
            {
                std::cerr << e.what() << std::endl;
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
}

void MainWindow::set_filter2d_n1(int n)
{
    if (!is_raw_mode())
    {
        cd_.filter2d_n1 = n;

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
}

void MainWindow::set_filter2d_n2(int n)
{
    if (!is_raw_mode())
    {
        cd_.filter2d_n2 = n;

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
}

void MainWindow::cancel_filter2d()
{
    if (!is_raw_mode())
    {
        if (cd_.filter2d_view_enabled == true)
            update_filter2d_view(false);
        pipe_refresh();
        notify();
    }
}

void MainWindow::set_fft_shift(const bool value)
{
    if (!is_raw_mode())
    {
        cd_.fft_shift_enabled = value;
        pipe_refresh();
    }
}

void MainWindow::set_time_transformation_size()
{
    if (!is_raw_mode())
    {
        int time_transformation_size = ui.timeTransformationSizeSpinBox->value();
        time_transformation_size = std::max(1, time_transformation_size);

        if (time_transformation_size == cd_.time_transformation_size)
            return;
        notify();
        auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());
        if (pipe)
        {
            pipe->insert_fn_end_vect([=]() {
                cd_.time_transformation_size = time_transformation_size;
                holovibes_.get_compute_pipe()->request_update_time_transformation_size();
                set_p_accu();
                // This will not do anything until
                // SliceWindow::changeTexture() isn't coded.
            });
        }
    }
}

void MainWindow::update_lens_view(bool value)
{
    cd_.gpu_lens_display_enabled = value;

    if (value)
    {
        try
        {
            // set positions of new windows according to the position of the
            // main GL window
            QPoint pos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 310, 0);
            ICompute* pipe = holovibes_.get_compute_pipe().get();

            const FrameDescriptor& fd = holovibes_.get_gpu_input_queue()->get_fd();
            ushort lens_window_width = fd.width;
            ushort lens_window_height = fd.height;
            get_good_size(lens_window_width, lens_window_height, auxiliary_window_max_size);

            lens_window.reset(new RawWindow(pos,
                                            QSize(lens_window_width, lens_window_height),
                                            pipe->get_lens_queue().get(),
                                            KindOfView::Lens));

            lens_window->setTitle("Lens view");
            lens_window->setCd(&cd_);

            // when the window is destoryed, disable_lens_view() will be triggered
            connect(lens_window.get(), SIGNAL(destroyed()), this, SLOT(disable_lens_view()));
        }
        catch (std::exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }

    else
    {
        disable_lens_view();
        lens_window.reset(nullptr);
    }

    pipe_refresh();
}

void MainWindow::disable_lens_view()
{
    if (lens_window)
        disconnect(lens_window.get(), SIGNAL(destroyed()), this, SLOT(disable_lens_view()));

    cd_.gpu_lens_display_enabled = false;
    holovibes_.get_compute_pipe()->request_disable_lens_view();
    notify();
}

void MainWindow::update_raw_view(bool value)
{
    if (value)
    {
        if (cd_.batch_size > global::global_config.output_queue_max_size)
        {
            ui.RawDisplayingCheckBox->setChecked(false);
            LOG_ERROR << "[RAW VIEW] Batch size must be lower than output queue size";
            return;
        }

        auto pipe = holovibes_.get_compute_pipe();
        pipe->request_raw_view();

        // Wait for the raw view to be enabled for notify
        while (pipe->get_raw_view_requested())
            continue;

        const FrameDescriptor& fd = holovibes_.get_gpu_input_queue()->get_fd();
        ushort raw_window_width = fd.width;
        ushort raw_window_height = fd.height;
        get_good_size(raw_window_width, raw_window_height, auxiliary_window_max_size);

        // set positions of new windows according to the position of the main GL
        // window and Lens window
        QPoint pos = mainDisplay->framePosition() + QPoint(mainDisplay->width() + 310, 0);
        raw_window.reset(
            new RawWindow(pos, QSize(raw_window_width, raw_window_height), pipe->get_raw_view_queue().get()));

        raw_window->setTitle("Raw view");
        raw_window->setCd(&cd_);

        connect(raw_window.get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));
    }
    else
    {
        raw_window.reset(nullptr);
        disable_raw_view();
    }
    pipe_refresh();
}

void MainWindow::disable_raw_view()
{
    if (raw_window)
        disconnect(raw_window.get(), SIGNAL(destroyed()), this, SLOT(disable_raw_view()));

    auto pipe = holovibes_.get_compute_pipe();
    pipe->request_disable_raw_view();

    // Wait for the raw view to be disabled for notify
    while (pipe->get_disable_raw_view_requested())
        continue;

    notify();
}

void MainWindow::set_p_accu()
{
    auto spinbox = ui.PAccSpinBox;
    auto checkBox = ui.PAccuCheckBox;
    cd_.p_accu_enabled = checkBox->isChecked();
    cd_.p_acc_level = spinbox->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_x_accu()
{
    auto box = ui.XAccSpinBox;
    auto checkBox = ui.XAccuCheckBox;
    cd_.x_accu_enabled = checkBox->isChecked();
    cd_.x_acc_level = box->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_y_accu()
{
    auto box = ui.YAccSpinBox;
    auto checkBox = ui.YAccuCheckBox;
    cd_.y_accu_enabled = checkBox->isChecked();
    cd_.y_acc_level = box->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_x_y()
{
    auto& fd = holovibes_.get_gpu_input_queue()->get_fd();
    uint x = ui.XSpinBox->value();
    uint y = ui.YSpinBox->value();

    if (x < fd.width)
        cd_.x_cuts = x;

    if (y < fd.height)
        cd_.y_cuts = y;
}

void MainWindow::set_q(int value)
{
    cd_.q_index = value;
    notify();
}

void MainWindow::set_q_acc()
{
    auto spinbox = ui.Q_AccSpinBox;
    auto checkBox = ui.Q_AccuCheckBox;
    cd_.q_acc_enabled = checkBox->isChecked();
    cd_.q_acc_level = spinbox->value();
    notify();
}

void MainWindow::set_p(int value)
{
    if (!is_raw_mode())
    {
        if (value < static_cast<int>(cd_.time_transformation_size))
        {
            cd_.pindex = value;
            pipe_refresh();
            notify();
        }
        else
            display_error("p param has to be between 1 and #img");
    }
}

void MainWindow::set_composite_intervals()
{
    // PRedSpinBox_Composite value cannont be higher than PBlueSpinBox_Composite
    ui.PRedSpinBox_Composite->setValue(std::min(ui.PRedSpinBox_Composite->value(), ui.PBlueSpinBox_Composite->value()));
    cd_.composite_p_red = ui.PRedSpinBox_Composite->value();
    cd_.composite_p_blue = ui.PBlueSpinBox_Composite->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_h_min()
{
    cd_.composite_p_min_h = ui.SpinBox_hue_freq_min->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_h_max()
{
    cd_.composite_p_max_h = ui.SpinBox_hue_freq_max->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_s_min()
{
    cd_.composite_p_min_s = ui.SpinBox_saturation_freq_min->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_s_max()
{
    cd_.composite_p_max_s = ui.SpinBox_saturation_freq_max->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_v_min()
{
    cd_.composite_p_min_v = ui.SpinBox_value_freq_min->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_intervals_hsv_v_max()
{
    cd_.composite_p_max_v = ui.SpinBox_value_freq_max->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_weights()
{
    cd_.weight_r = ui.WeightSpinBox_R->value();
    cd_.weight_g = ui.WeightSpinBox_G->value();
    cd_.weight_b = ui.WeightSpinBox_B->value();
    pipe_refresh();
    notify();
}

void MainWindow::set_composite_auto_weights(bool value)
{
    cd_.composite_auto_weights_ = value;
    set_auto_contrast();
}

void MainWindow::click_composite_rgb_or_hsv()
{
    cd_.composite_kind = ui.radioButton_rgb->isChecked() ? CompositeKind::RGB : CompositeKind::HSV;
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
    cd_.composite_p_activated_s = ui.checkBox_saturation_freq->isChecked();
    ui.SpinBox_saturation_freq_min->setDisabled(!ui.checkBox_saturation_freq->isChecked());
    ui.SpinBox_saturation_freq_max->setDisabled(!ui.checkBox_saturation_freq->isChecked());
}

void MainWindow::actualize_frequency_channel_v()
{
    cd_.composite_p_activated_v = ui.checkBox_value_freq->isChecked();
    ui.SpinBox_value_freq_min->setDisabled(!ui.checkBox_value_freq->isChecked());
    ui.SpinBox_value_freq_max->setDisabled(!ui.checkBox_value_freq->isChecked());
}

void MainWindow::actualize_checkbox_h_gaussian_blur()
{
    cd_.h_blur_activated = ui.checkBox_h_gaussian_blur->isChecked();
    ui.SpinBox_hue_blur_kernel_size->setEnabled(ui.checkBox_h_gaussian_blur->isChecked());
}

void MainWindow::actualize_kernel_size_blur() { cd_.h_blur_kernel_size = ui.SpinBox_hue_blur_kernel_size->value(); }

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
    // Store the slider value in cd_ (ComputeDescriptor)
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
    slide_update_threshold(*ui.horizontalSlider_hue_threshold_min,
                           cd_.slider_h_threshold_min,
                           cd_.slider_h_threshold_max,
                           *ui.horizontalSlider_hue_threshold_max,
                           *ui.label_hue_threshold_min,
                           cd_.slider_h_threshold_min,
                           cd_.slider_h_threshold_max);
}

void MainWindow::slide_update_threshold_h_max()
{
    slide_update_threshold(*ui.horizontalSlider_hue_threshold_max,
                           cd_.slider_h_threshold_max,
                           cd_.slider_h_threshold_min,
                           *ui.horizontalSlider_hue_threshold_min,
                           *ui.label_hue_threshold_max,
                           cd_.slider_h_threshold_min,
                           cd_.slider_h_threshold_max);
}

void MainWindow::slide_update_threshold_s_min()
{
    slide_update_threshold(*ui.horizontalSlider_saturation_threshold_min,
                           cd_.slider_s_threshold_min,
                           cd_.slider_s_threshold_max,
                           *ui.horizontalSlider_saturation_threshold_max,
                           *ui.label_saturation_threshold_min,
                           cd_.slider_s_threshold_min,
                           cd_.slider_s_threshold_max);
}

void MainWindow::slide_update_threshold_s_max()
{
    slide_update_threshold(*ui.horizontalSlider_saturation_threshold_max,
                           cd_.slider_s_threshold_max,
                           cd_.slider_s_threshold_min,
                           *ui.horizontalSlider_saturation_threshold_min,
                           *ui.label_saturation_threshold_max,
                           cd_.slider_s_threshold_min,
                           cd_.slider_s_threshold_max);
}

void MainWindow::slide_update_threshold_v_min()
{
    slide_update_threshold(*ui.horizontalSlider_value_threshold_min,
                           cd_.slider_v_threshold_min,
                           cd_.slider_v_threshold_max,
                           *ui.horizontalSlider_value_threshold_max,
                           *ui.label_value_threshold_min,
                           cd_.slider_v_threshold_min,
                           cd_.slider_v_threshold_max);
}

void MainWindow::slide_update_threshold_v_max()
{
    slide_update_threshold(*ui.horizontalSlider_value_threshold_max,
                           cd_.slider_v_threshold_max,
                           cd_.slider_v_threshold_min,
                           *ui.horizontalSlider_value_threshold_min,
                           *ui.label_value_threshold_max,
                           cd_.slider_v_threshold_min,
                           cd_.slider_v_threshold_max);
}

void MainWindow::increment_p()
{
    if (!is_raw_mode())
    {

        if (cd_.pindex < cd_.time_transformation_size)
        {
            cd_.pindex = cd_.pindex + 1;
            set_auto_contrast();
            notify();
        }
        else
            display_error("p param has to be between 1 and #img");
    }
}

void MainWindow::decrement_p()
{
    if (!is_raw_mode())
    {
        if (cd_.pindex > 0)
        {
            cd_.pindex = cd_.pindex - 1;
            set_auto_contrast();
            notify();
        }
        else
            display_error("p param has to be between 1 and #img");
    }
}

void MainWindow::set_wavelength(const double value)
{
    if (!is_raw_mode())
    {
        cd_.lambda = static_cast<float>(value) * 1.0e-9f;
        pipe_refresh();
    }
}

void MainWindow::set_z(const double value)
{
    if (!is_raw_mode())
    {
        cd_.zdistance = static_cast<float>(value);
        pipe_refresh();
    }
}

void MainWindow::increment_z()
{
    if (!is_raw_mode())
    {
        set_z(cd_.zdistance + z_step_);
        ui.ZDoubleSpinBox->setValue(cd_.zdistance);
    }
}

void MainWindow::decrement_z()
{
    if (!is_raw_mode())
    {
        set_z(cd_.zdistance - z_step_);
        ui.ZDoubleSpinBox->setValue(cd_.zdistance);
    }
}

void MainWindow::set_z_step(const double value)
{
    z_step_ = value;
    ui.ZDoubleSpinBox->setSingleStep(value);
}

void MainWindow::set_space_transformation(const QString value)
{
    if (!is_raw_mode())
    {
        if (value == "None")
            cd_.space_transformation = SpaceTransformation::None;
        else if (value == "1FFT")
            cd_.space_transformation = SpaceTransformation::FFT1;
        else if (value == "2FFT")
            cd_.space_transformation = SpaceTransformation::FFT2;
        else
        {
            // Shouldn't happen
            cd_.space_transformation = SpaceTransformation::None;
            LOG_ERROR << "Unknown space transform: " << value.toStdString() << ", falling back to None";
        }
        set_holographic_mode();
    }
}

void MainWindow::set_time_transformation(QString value)
{
    if (!is_raw_mode())
    {
        if (value == "STFT")
            cd_.time_transformation = TimeTransformation::STFT;
        else if (value == "PCA")
            cd_.time_transformation = TimeTransformation::PCA;
        else if (value == "None")
            cd_.time_transformation = TimeTransformation::NONE;
        else if (value == "SSA_STFT")
            cd_.time_transformation = TimeTransformation::SSA_STFT;
        set_holographic_mode();
    }
}

void MainWindow::set_unwrapping_2d(const bool value)
{
    if (!is_raw_mode())
    {
        holovibes_.get_compute_pipe()->request_unwrapping_2d(value);
        pipe_refresh();
        notify();
    }
}

void MainWindow::set_accumulation(bool value)
{
    if (!is_raw_mode())
    {
        cd_.set_accumulation(cd_.current_window, value);
        pipe_refresh();
        notify();
    }
}

void MainWindow::set_accumulation_level(int value)
{
    if (!is_raw_mode())
    {
        cd_.set_accumulation_level(cd_.current_window, value);
        pipe_refresh();
    }
}

void MainWindow::pipe_refresh()
{
    if (!is_raw_mode())
    {
        try
        {
            // FIXME: Should better not use a if structure with 2 method access, 1 dereferencing, and 1 negation bitwise
            // operation to set a boolean
            // But maybe a simple read access that create a false condition result is better than simply making a
            // writting access
            if (!holovibes_.get_compute_pipe()->get_request_refresh())
                holovibes_.get_compute_pipe()->request_refresh();
        }
        catch (std::runtime_error& e)
        {
        }
    }
}

void MainWindow::set_composite_area() { mainDisplay->getOverlayManager().create_overlay<CompositeArea>(); }

#pragma endregion
/* ------------ */
#pragma region Texture
void MainWindow::rotateTexture()
{
    WindowKind curWin = cd_.current_window;

    if (curWin == WindowKind::XYview)
    {
        displayAngle = (displayAngle == 270.f) ? 0.f : displayAngle + 90.f;
        mainDisplay->setAngle(displayAngle);
    }
    else if (sliceXZ && curWin == WindowKind::XZview)
    {
        xzAngle = (xzAngle == 270.f) ? 0.f : xzAngle + 90.f;
        sliceXZ->setAngle(xzAngle);
    }
    else if (sliceYZ && curWin == WindowKind::YZview)
    {
        yzAngle = (yzAngle == 270.f) ? 0.f : yzAngle + 90.f;
        sliceYZ->setAngle(yzAngle);
    }
    notify();
}

void MainWindow::flipTexture()
{
    WindowKind curWin = cd_.current_window;

    if (curWin == WindowKind::XYview)
    {
        displayFlip = !displayFlip;
        mainDisplay->setFlip(displayFlip);
    }
    else if (sliceXZ && curWin == WindowKind::XZview)
    {
        xzFlip = !xzFlip;
        sliceXZ->setFlip(xzFlip);
    }
    else if (sliceYZ && curWin == WindowKind::YZview)
    {
        yzFlip = !yzFlip;
        sliceYZ->setFlip(yzFlip);
    }
    notify();
}

#pragma endregion
/* ------------ */
#pragma region Contrast - Log
void MainWindow::set_contrast_mode(bool value)
{
    if (!is_raw_mode())
    {
        change_window();
        cd_.contrast_enabled = value;
        cd_.contrast_auto_refresh = true;
        pipe_refresh();
        notify();
    }
}

void MainWindow::set_auto_contrast_cuts()
{
    if (auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get()))
    {
        pipe->autocontrast_end_pipe(WindowKind::XZview);
        pipe->autocontrast_end_pipe(WindowKind::YZview);
    }
}

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

void MainWindow::set_auto_contrast()
{
    if (!is_raw_mode())
    {
        try
        {
            if (auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get()))
                pipe->autocontrast_end_pipe(cd_.current_window);
        }
        catch (std::runtime_error& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
}

void MainWindow::set_contrast_min(const double value)
{
    if (!is_raw_mode())
    {
        if (cd_.contrast_enabled)
        {
            // FIXME: type issue, manipulatiion of double casted to float implies lost of data
            // Get the minimum contrast value rounded for the comparison
            const float old_val = cd_.get_truncate_contrast_min(cd_.current_window);
            // Floating number issue: cast to float for the comparison
            const float val = value;
            if (old_val != val)
            {
                cd_.set_contrast_min(cd_.current_window, value);
                pipe_refresh();
            }
        }
    }
}

void MainWindow::set_contrast_max(const double value)
{
    if (!is_raw_mode())
    {
        if (cd_.contrast_enabled)
        {
            // FIXME: type issue, manipulatiion of double casted to float implies lost of data
            // Get the maximum contrast value rounded for the comparison
            const float old_val = cd_.get_truncate_contrast_max(cd_.current_window);
            // Floating number issue: cast to float for the comparison
            const float val = value;
            if (old_val != val)
            {
                cd_.set_contrast_max(cd_.current_window, value);
                pipe_refresh();
            }
        }
    }
}

void MainWindow::invert_contrast(bool value)
{
    if (!is_raw_mode())
    {
        if (cd_.contrast_enabled)
        {
            cd_.contrast_invert = value;
            pipe_refresh();
        }
    }
}

void MainWindow::set_auto_refresh_contrast(bool value)
{
    cd_.contrast_auto_refresh = value;
    pipe_refresh();
    notify();
}

void MainWindow::set_log_scale(const bool value)
{
    if (!is_raw_mode())
    {
        cd_.set_log_scale_slice_enabled(cd_.current_window, value);
        if (cd_.contrast_enabled && value)
            set_auto_contrast();
        pipe_refresh();
        notify();
    }
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
        catch (const std::exception&)
        {
            cd_.convolution_enabled = false;
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
    catch (const std::exception&)
    {
        cd_.convolution_enabled = false;
    }

    notify();
}

void MainWindow::set_divide_convolution_mode(const bool value)
{
    cd_.divide_convolution_enabled = value;

    pipe_refresh();
    notify();
}

void MainWindow::set_fast_pipe(bool value)
{
    auto pipe = dynamic_cast<Pipe*>(holovibes_.get_compute_pipe().get());
    if (pipe && value)
    {
        pipe->insert_fn_end_vect([=]() {
            // Constraints linked with fast pipe option
            cd_.time_transformation_stride = cd_.batch_size.load();
            cd_.time_transformation_size = cd_.batch_size.load();
            pipe->request_update_time_transformation_stride();
            pipe->request_update_time_transformation_size();
            cd_.fast_pipe = true;
            pipe_refresh();
            notify();
        });
    }
    else
    {
        cd_.fast_pipe = false;
        pipe_refresh();
        notify();
    }
}

#pragma endregion
/* ------------ */
#pragma region Reticle
void MainWindow::display_reticle(bool value)
{
    cd_.reticle_enabled = value;
    if (value)
    {
        mainDisplay->getOverlayManager().create_overlay<Reticle>();
        mainDisplay->getOverlayManager().create_default();
    }
    else
    {
        mainDisplay->getOverlayManager().disable_all(Reticle);
    }
    pipe_refresh();
    notify();
}

void MainWindow::reticle_scale(double value)
{
    if (0 > value || value > 1)
        return;

    cd_.reticle_scale = value;
    pipe_refresh();
}
#pragma endregion Reticle
/* ------------ */
#pragma region Chart
void MainWindow::activeSignalZone()
{
    mainDisplay->getOverlayManager().create_overlay<Signal>();
    notify();
}

void MainWindow::activeNoiseZone()
{
    mainDisplay->getOverlayManager().create_overlay<Noise>();
    notify();
}

void MainWindow::start_chart_display()
{
    if (cd_.chart_display_enabled)
        return;

    auto pipe = holovibes_.get_compute_pipe();
    pipe->request_display_chart();

    // Wait for the chart display to be enabled for notify
    while (pipe->get_chart_display_requested())
        continue;

    plot_window_ = std::make_unique<PlotWindow>(*holovibes_.get_compute_pipe()->get_chart_display_queue(),
                                                auto_scale_point_threshold_,
                                                "Chart");
    connect(plot_window_.get(), SIGNAL(closed()), this, SLOT(stop_chart_display()), Qt::UniqueConnection);

    ui.ChartPlotPushButton->setEnabled(false);
}

void MainWindow::stop_chart_display()
{
    if (!cd_.chart_display_enabled)
        return;

    try
    {
        auto pipe = holovibes_.get_compute_pipe();
        pipe->request_disable_display_chart();

        // Wait for the chart display to be disabled for notify
        while (pipe->get_disable_chart_display_requested())
            continue;
    }
    catch (const std::exception&)
    {
    }

    plot_window_.reset(nullptr);

    ui.ChartPlotPushButton->setEnabled(true);
}
#pragma endregion
/* ------------ */
#pragma region Record
void MainWindow::set_record_frame_step(int value)
{
    record_frame_step_ = value;
    ui.NumberOfFramesSpinBox->setSingleStep(value);
}

void MainWindow::set_nb_frames_mode(bool value) { ui.NumberOfFramesSpinBox->setEnabled(value); }

void MainWindow::browse_record_output_file()
{
    QString filepath;

    // Open file explorer dialog on the fly depending on the record mode
    if (record_mode_ == RecordMode::CHART)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Chart output file"),
                                                record_output_directory_.c_str(),
                                                tr("Text files (*.txt);;CSV files (*.csv)"));
    }
    else if (record_mode_ == RecordMode::RAW)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                record_output_directory_.c_str(),
                                                tr("Holo files (*.holo)"));
    }
    else if (record_mode_ == RecordMode::HOLOGRAM)
    {
        filepath = QFileDialog::getSaveFileName(this,
                                                tr("Record output file"),
                                                record_output_directory_.c_str(),
                                                tr("Holo files (*.holo);; Avi Files (*.avi);; Mp4 files (*.mp4)"));
    }

    if (filepath.isEmpty())
        return;

    // Convert QString to std::string
    std::string std_filepath = filepath.toStdString();

    std::replace(std_filepath.begin(), std_filepath.end(), '/', '\\');
    std::filesystem::path path = std::filesystem::path(std_filepath);

    record_output_directory_ = path.parent_path().string();
    /*  cppreference: https://en.cppreference.com/w/cpp/filesystem/path/extension
     *  -> rightmost ".*":
     *     std::filesystem::path("/foo/bar.mp4.holo.mp4").extension() -> ".mp4"
     */
    const std::string file_ext = path.extension().string();
    std::string filename = path.filename().string();

    // Get the first file_ext string position in filename
    /* cppreference: https://www.cplusplus.com/reference/string/string/find/
     *   std::string(/foo/bar.mp4.holo.mp4).find(".mp4") -> 8 corresponding to : /foo/bar(.mp4).holo.mp4
     *
     *  FIXME: Could be an unexpected behaviour
     *  To conclude, the extension you get with "file_ext = path.extension().string()" is not always
     *  the string you locate in "filename.find(file_ext)"
     */
    std::size_t ext_pos = filename.find(file_ext);
    if (ext_pos != std::string::npos)
        // if file_ext not found in filename
        filename.erase(ext_pos, file_ext.length());

    ui.RecordExtComboBox->setCurrentText(file_ext.c_str());

    default_output_filename_ = filename;

    notify();
}

void MainWindow::browse_batch_input()
{

    // Open file explorer on the fly
    QString filename =
        QFileDialog::getOpenFileName(this, tr("Batch input file"), batch_input_directory_.c_str(), tr("All files (*)"));

    // Output the file selected in he ui line edit widget
    QLineEdit* batch_input_line_edit = ui.BatchInputPathLineEdit;
    batch_input_line_edit->clear();
    batch_input_line_edit->insert(filename);
}

void MainWindow::set_record_mode(const QString& value)
{
    if (record_mode_ == RecordMode::CHART)
        stop_chart_display();

    stop_record();

    const std::string text = value.toUtf8();

    if (text == "Chart")
        record_mode_ = RecordMode::CHART;
    else if (text == "Processed Image")
        record_mode_ = RecordMode::HOLOGRAM;
    else if (text == "Raw Image")
        record_mode_ = RecordMode::RAW;
    else
        throw std::exception("Record mode not handled");

    if (record_mode_ == RecordMode::CHART)
    {
        ui.RecordExtComboBox->clear();
        ui.RecordExtComboBox->insertItem(0, ".csv");
        ui.RecordExtComboBox->insertItem(1, ".txt");

        ui.ChartPlotWidget->show();

        if (mainDisplay)
        {
            mainDisplay->resetTransform();

            mainDisplay->getOverlayManager().enable_all(Signal);
            mainDisplay->getOverlayManager().enable_all(Noise);
            mainDisplay->getOverlayManager().create_overlay<Signal>();
        }
    }
    else
    {
        if (record_mode_ == RecordMode::RAW)
        {
            ui.RecordExtComboBox->clear();
            ui.RecordExtComboBox->insertItem(0, ".holo");
        }
        else if (record_mode_ == RecordMode::HOLOGRAM)
        {
            ui.RecordExtComboBox->clear();
            ui.RecordExtComboBox->insertItem(0, ".holo");
            ui.RecordExtComboBox->insertItem(1, ".avi");
            ui.RecordExtComboBox->insertItem(2, ".mp4");
        }

        ui.ChartPlotWidget->hide();

        if (mainDisplay)
        {
            mainDisplay->resetTransform();

            mainDisplay->getOverlayManager().disable_all(Signal);
            mainDisplay->getOverlayManager().disable_all(Noise);
        }
    }

    notify();
}

void MainWindow::stop_record()
{
    holovibes_.stop_batch_gpib();

    if (record_mode_ == RecordMode::CHART)
        holovibes_.stop_chart_record();
    else if (record_mode_ == RecordMode::HOLOGRAM || record_mode_ == RecordMode::RAW)
        holovibes_.stop_frame_record();
}

void MainWindow::record_finished(RecordMode record_mode)
{
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
    ui.BatchSizeSpinBox->setEnabled(cd_.compute_mode == Computation::Hologram);
    is_recording_ = false;
}

void MainWindow::start_record()
{
    bool batch_enabled = ui.BatchGroupBox->isChecked();

    // Preconditions to start record

    std::optional<unsigned int> nb_frames_to_record = ui.NumberOfFramesSpinBox->value();
    if (!ui.NumberOfFramesCheckBox->isChecked())
        nb_frames_to_record = std::nullopt;

    if ((record_mode_ == RecordMode::CHART || batch_enabled) && nb_frames_to_record == std::nullopt)
        return display_error("Number of frames must be activated");

    std::string output_path =
        ui.OutputFilePathLineEdit->text().toStdString() + ui.RecordExtComboBox->currentText().toStdString();

    std::string batch_input_path = ui.BatchInputPathLineEdit->text().toUtf8();
    if (batch_enabled && batch_input_path == "")
        return display_error("No batch input file");

    // Start record

    raw_window.reset(nullptr);
    disable_raw_view();
    ui.RawDisplayingCheckBox->setHidden(true);

    ui.BatchSizeSpinBox->setEnabled(false);
    is_recording_ = true;

    ui.ExportRecPushButton->setEnabled(false);
    ui.ExportStopPushButton->setEnabled(true);

    ui.RecordProgressBar->reset();
    ui.RecordProgressBar->show();

    auto callback = [record_mode = record_mode_, this]() {
        synchronize_thread([=]() { record_finished(record_mode); });
    };

    if (batch_enabled)
    {
        holovibes_.start_batch_gpib(batch_input_path,
                                    output_path,
                                    nb_frames_to_record.value(),
                                    record_mode_,
                                    square_output,
                                    callback);
    }
    else
    {
        if (record_mode_ == RecordMode::CHART)
        {
            holovibes_.start_chart_record(output_path, nb_frames_to_record.value(), callback);
        }
        else if (record_mode_ == RecordMode::HOLOGRAM)
        {
            holovibes_.start_frame_record(output_path, nb_frames_to_record, false, square_output, 0, callback);
        }
        else if (record_mode_ == RecordMode::RAW)
        {
            holovibes_.start_frame_record(output_path, nb_frames_to_record, true, false, 0, callback);
        }
    }
}
#pragma endregion
/* ------------ */
#pragma region Import
void MainWindow::set_start_stop_buttons(bool value)
{
    ui.ImportStartPushButton->setEnabled(value);
    ui.ImportStopPushButton->setEnabled(value);
}

void MainWindow::import_browse_file()
{
    QString filename = "";

    // Open the file explorer to let the user pick his file
    // and store the chosen file in filename
    filename = QFileDialog::getOpenFileName(this,
                                            tr("import file"),
                                            file_input_directory_.c_str(),
                                            tr("All files (*.holo *.cine);; Holo files (*.holo);; Cine files "
                                               "(*.cine)"));

    // Start importing the chosen
    import_file(filename);
}

void MainWindow::import_file(const QString& filename)
{
    // Get the widget (output bar) from the ui linked to the file explorer
    QLineEdit* import_line_edit = ui.ImportPathLineEdit;
    // Insert the newly getted path in it
    import_line_edit->clear();
    import_line_edit->insert(filename);

    if (filename != "")
    {
        try
        {
            // Will throw if the file format (extension) cannot be handled
            io_files::InputFrameFile* input_file = io_files::InputFrameFileFactory::open(filename.toStdString());

            // Gather data from the newly opened file
            size_t nb_frames = input_file->get_total_nb_frames();
            file_fd_ = input_file->get_frame_descriptor();
            input_file->import_compute_settings(cd_);

            // Don't need the input file anymore
            delete input_file;

            // Update the ui with the gathered data
            ui.ImportEndIndexSpinBox->setMaximum(nb_frames);
            ui.ImportEndIndexSpinBox->setValue(nb_frames);

            // We can now launch holovibes over this file
            set_start_stop_buttons(true);
        }
        catch (const io_files::FileException& e)
        {
            // In case of bad format, we triggered the user
            QMessageBox messageBox;
            messageBox.critical(nullptr, "File Error", e.what());

            // Holovibes cannot be launched over this file
            set_start_stop_buttons(false);
        }
    }

    else
        set_start_stop_buttons(false);
}

void MainWindow::import_stop()
{
    close_windows();
    cancel_time_transformation_cuts();

    holovibes_.stop_all_worker_controller();
    holovibes_.start_information_display(false);

    close_critical_compute();

    // FIXME: import_stop() and camera_none() call same methods
    // FIXME: camera_none() weird call because we are dealing with imported file
    camera_none();

    cd_.is_computation_stopped = true;

    notify();
}

void MainWindow::import_start()
{
    // shift main window when camera view appears
    QRect rec = QGuiApplication::primaryScreen()->geometry();
    int screen_height = rec.height();
    int screen_width = rec.width();
    move(QPoint(210 + (screen_width - 800) / 2, 200 + (screen_height - 500) / 2));

    if (!cd_.is_computation_stopped)
        // if computation is running
        import_stop();

    cd_.is_computation_stopped = false;
    // Gather all the usefull data from the ui import panel
    init_holovibes_import_mode();

    ui.ImageModeComboBox->setCurrentIndex(is_raw_mode() ? 0 : 1);
}

void MainWindow::init_holovibes_import_mode()
{
    // Get all the useful ui items
    QLineEdit* import_line_edit = ui.ImportPathLineEdit;
    QSpinBox* fps_spinbox = ui.ImportInputFpsSpinBox;
    QSpinBox* start_spinbox = ui.ImportStartIndexSpinBox;
    QCheckBox* load_file_gpu_box = ui.LoadFileInGpuCheckBox;
    QSpinBox* end_spinbox = ui.ImportEndIndexSpinBox;

    // Set the image rendering ui params
    cd_.time_transformation_stride = std::ceil(static_cast<float>(fps_spinbox->value()) / 20.0f);
    cd_.batch_size = 1;

    // Because we are in import mode
    is_enabled_camera_ = false;

    try
    {
        // Gather data from import panel
        std::string file_path = import_line_edit->text().toUtf8();
        unsigned int fps = fps_spinbox->value();
        size_t first_frame = start_spinbox->value();
        size_t last_frame = end_spinbox->value();
        bool load_file_in_gpu = load_file_gpu_box->isChecked();

        holovibes_.init_input_queue(file_fd_);
        holovibes_.start_file_frame_read(file_path,
                                         true,
                                         fps,
                                         first_frame - 1,
                                         last_frame - first_frame + 1,
                                         load_file_in_gpu,
                                         [=]() {
                                             synchronize_thread([&]() {
                                                 if (cd_.is_computation_stopped)
                                                     ui.FileReaderProgressBar->hide();
                                             });
                                         });
        ui.FileReaderProgressBar->show();
    }
    catch (std::exception& e)
    {
        display_error(e.what());
        is_enabled_camera_ = false;
        mainDisplay.reset(nullptr);
        holovibes_.stop_compute();
        holovibes_.stop_frame_read();
        return;
    }

    is_enabled_camera_ = true;
    set_image_mode(nullptr);

    // Make camera's settings menu unaccessible
    QAction* settings = ui.actionSettings;
    settings->setEnabled(false);

    import_type_ = ImportType::File;

    notify();
}

void MainWindow::import_start_spinbox_update()
{
    QSpinBox* start_spinbox = ui.ImportStartIndexSpinBox;
    QSpinBox* end_spinbox = ui.ImportEndIndexSpinBox;

    if (start_spinbox->value() > end_spinbox->value())
        end_spinbox->setValue(start_spinbox->value());
}

void MainWindow::import_end_spinbox_update()
{
    QSpinBox* start_spinbox = ui.ImportStartIndexSpinBox;
    QSpinBox* end_spinbox = ui.ImportEndIndexSpinBox;

    if (end_spinbox->value() < start_spinbox->value())
        start_spinbox->setValue(end_spinbox->value());
}

#pragma endregion

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
    auto lambda = [this, n]() { ui.FileReaderProgressBar->setValue(n); };
    synchronize_thread(lambda);
}
#pragma endregion
} // namespace gui
} // namespace holovibes
