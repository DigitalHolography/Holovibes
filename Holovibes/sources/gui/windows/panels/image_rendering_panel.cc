/*! \file
 *
 */

#include <filesystem>

#include "image_rendering_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"
#include "API.hh"
#include "GUI.hh"
#include "user_interface_descriptor.hh"

#include <map>

namespace holovibes::gui
{
ImageRenderingPanel::ImageRenderingPanel(QWidget* parent)
    : Panel(parent)
{
    z_up_shortcut_ = new QShortcut(QKeySequence("Up"), this);
    z_up_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(z_up_shortcut_, SIGNAL(activated()), this, SLOT(increment_z()));

    z_down_shortcut_ = new QShortcut(QKeySequence("Down"), this);
    z_down_shortcut_->setContext(Qt::ApplicationShortcut);
    connect(z_down_shortcut_, SIGNAL(activated()), this, SLOT(decrement_z()));
}

ImageRenderingPanel::~ImageRenderingPanel()
{
    delete z_up_shortcut_;
    delete z_down_shortcut_;
}

void ImageRenderingPanel::init() { ui_->ZDoubleSpinBox->setSingleStep(z_step_); }

void ImageRenderingPanel::on_notify()
{
    const bool is_raw = api_.compute.get_compute_mode() == Computation::Raw;
    const bool is_data_not_moments = !(api_.input.get_data_type() == RecordedDataType::MOMENTS);
    const bool not_raw_not_moments = !is_raw && is_data_not_moments;

    ui_->ImageModeComboBox->setCurrentIndex(static_cast<int>(api_.compute.get_compute_mode()));
    ui_->ImageModeComboBox->setEnabled(is_data_not_moments);

    ui_->TimeStrideSpinBox->setEnabled(!is_raw);

    ui_->TimeStrideSpinBox->setValue(api_.transform.get_time_stride());
    ui_->TimeStrideSpinBox->setSingleStep(api_.transform.get_batch_size());
    ui_->TimeStrideSpinBox->setMinimum(api_.transform.get_batch_size());

    const bool is_batch_size_enabled = !api_.record.is_recording() && is_data_not_moments;
    ui_->BatchSizeSpinBox->setValue(api_.transform.get_batch_size());
    ui_->BatchSizeSpinBox->setEnabled(is_batch_size_enabled);
    ui_->BatchSizeLabel->setEnabled(is_batch_size_enabled);

    ui_->BatchSizeSpinBox->setMaximum(api_.input.get_input_buffer_size());

    ui_->SpaceTransformationLabel->setEnabled(not_raw_not_moments);
    ui_->SpaceTransformationComboBox->setEnabled(not_raw_not_moments);
    ui_->SpaceTransformationComboBox->setCurrentIndex(static_cast<int>(api_.transform.get_space_transformation()));
    ui_->TimeTransformationLabel->setEnabled(not_raw_not_moments);
    ui_->TimeTransformationComboBox->setEnabled(not_raw_not_moments);
    ui_->TimeTransformationComboBox->setCurrentIndex(static_cast<int>(api_.transform.get_time_transformation()));

    // Changing time_transformation_size with time transformation cuts is
    // supported by the pipe, but some modifications have to be done in
    // SliceWindow, OpenGl buffers.
    ui_->timeTransformationSizeSpinBox->setEnabled(!is_raw && !api_.view.get_cuts_view_enabled());
    ui_->timeTransformationSizeSpinBox->setValue(api_.transform.get_time_transformation_size());

    // Z (focus)
    ui_->LambdaLabel->setEnabled(not_raw_not_moments);
    ui_->LambdaSpinBox->setEnabled(not_raw_not_moments);
    ui_->LambdaSpinBox->setValue(api_.transform.get_lambda() * 1.0e9f);
    ui_->ZLabel->setEnabled(not_raw_not_moments);
    ui_->ZDoubleSpinBox->setEnabled(not_raw_not_moments);
    ui_->ZDoubleSpinBox->setValue(api_.transform.get_z_distance() * 1000);
    ui_->ZDoubleSpinBox->setSingleStep(z_step_);
    ui_->ZSlider->setEnabled(not_raw_not_moments);
    ui_->BoundaryDoubleSpinBox->setValue(api_.information.get_boundary() * 1000);

    // Filter2D
    bool filter2D_enabled = !is_raw && api_.filter2d.get_filter2d_enabled();
    ui_->Filter2D->setEnabled(!is_raw);
    ui_->Filter2D->setChecked(filter2D_enabled);

    ui_->Filter2DView->setVisible(filter2D_enabled);
    ui_->Filter2DView->setChecked(!is_raw && api_.view.get_filter2d_view_enabled());
    ui_->Filter2DN1SpinBox->setVisible(filter2D_enabled);
    ui_->Filter2DN1SpinBox->setValue(api_.filter2d.get_filter2d_n1());

    ui_->Filter2DN2SpinBox->setVisible(filter2D_enabled);
    ui_->Filter2DN2SpinBox->setValue(api_.filter2d.get_filter2d_n2());
    ui_->Filter2DN1SpinBox->setMaximum(ui_->Filter2DN2SpinBox->value() - 1);

    // Filter
    ui_->InputFilterLabel->setVisible(filter2D_enabled);
    ui_->InputFilterQuickSelectComboBox->setVisible(filter2D_enabled);
    int index = 0;
    if (!api_.filter2d.get_filter_file_name().empty())
        index =
            ui_->InputFilterQuickSelectComboBox->findText(QString::fromStdString(api_.filter2d.get_filter_file_name()));

    ui_->InputFilterQuickSelectComboBox->setCurrentIndex(index);

    // Convolution
    ui_->ConvoCheckBox->setVisible(api_.compute.get_compute_mode() == Computation::Hologram);
    ui_->ConvoCheckBox->setChecked(api_.global_pp.get_convolution_enabled());

    ui_->DivideConvoCheckBox->setVisible(api_.global_pp.get_convolution_enabled());
    ui_->DivideConvoCheckBox->setChecked(api_.global_pp.get_divide_convolution_enabled());
    ui_->KernelQuickSelectComboBox->setVisible(api_.global_pp.get_convolution_enabled());

    index = 0;
    if (!api_.global_pp.get_convolution_file_name().empty())
        index = ui_->KernelQuickSelectComboBox->findText(
            QString::fromStdString(api_.global_pp.get_convolution_file_name()));

    ui_->KernelQuickSelectComboBox->setCurrentIndex(index);
}

void ImageRenderingPanel::load_gui(const json& j_us)
{
    z_step_ = json_get_or_default(j_us, z_step_, "gui settings", "z step");
    bool h = json_get_or_default(j_us, isHidden(), "panels", "image rendering hidden");
    ui_->actionImage_rendering->setChecked(!h);
    setHidden(h);
}

void ImageRenderingPanel::save_gui(json& j_us)
{
    j_us["gui settings"]["z step"] = z_step_;
    j_us["panels"]["image rendering hidden"] = isHidden();
}

void ImageRenderingPanel::set_compute_mode(int mode)
{
    Computation comp_mode = static_cast<Computation>(mode);

    gui::close_windows();
    if (api_.compute.set_compute_mode(comp_mode) != ApiCode::OK)
        return;

    if (!api_.compute.get_is_computation_stopped())
        gui::create_window(comp_mode, parent_->window_max_size);

    if (comp_mode == Computation::Hologram)
    {
        /* Filter2D */
        camera::FrameDescriptor fd = api_.input.get_input_fd();
        ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));
    }

    parent_->notify();
}

void ImageRenderingPanel::update_batch_size()
{
    if (api_.transform.set_batch_size(ui_->BatchSizeSpinBox->value()) == ApiCode::OK)
        parent_->notify();
}

void ImageRenderingPanel::update_time_stride()
{
    if (api_.transform.set_time_stride(ui_->TimeStrideSpinBox->value()) == ApiCode::OK)
        parent_->notify();
}

void ImageRenderingPanel::set_filter2d(bool checked)
{
    if (api_.compute.get_compute_mode() == Computation::Raw)
        return;

    api_.filter2d.set_filter2d_enabled(checked);

    if (checked)
    {
        // Set the input box related to the filter2d
        const camera::FrameDescriptor& fd = api_.input.get_input_fd();
        const int size_max = floor((fmax(fd.width, fd.height) / 2) * M_SQRT2);
        ui_->Filter2DN2SpinBox->setMaximum(size_max);
        // sets the filter_2d_n2 so the frame fits in the lens diameter by default
        api_.filter2d.set_filter2d_n2(size_max);
        ui_->Filter2DN2SpinBox->setValue(size_max);
    }
    else
        update_filter2d_view(false);

    parent_->notify();
}

void ImageRenderingPanel::set_filter2d_n1(int n) { api_.filter2d.set_filter2d_n1(n); }

void ImageRenderingPanel::set_filter2d_n2(int n)
{
    ui_->Filter2DN1SpinBox->setMaximum(n - 1);
    api_.filter2d.set_filter2d_n2(n);
}

void ImageRenderingPanel::update_input_filter(const QString& value)
{
    LOG_FUNC();

    std::string v = value.toStdString();
    api_.filter2d.enable_filter(v == UID_FILTER_TYPE_DEFAULT ? "" : v);
}

void ImageRenderingPanel::update_filter2d_view(bool checked)
{
    api_.view.set_filter2d_view(checked);
    gui::set_filter2d_view(checked, parent_->auxiliary_window_max_size);
    parent_->notify();
}

void ImageRenderingPanel::set_space_transformation(const QString& value)
{
    SpaceTransformation st;

    try
    {
        // json{} return an array
        st = json{value.toStdString()}[0].get<SpaceTransformation>();
        LOG_DEBUG("value.toStdString() : {}", value.toStdString());
    }
    catch (std::out_of_range& e)
    {
        LOG_ERROR("Catch {}", e.what());
        throw;
    }

    if (api_.transform.set_space_transformation(st) == ApiCode::OK)
        parent_->notify();
}

void ImageRenderingPanel::set_time_transformation(const QString& value)
{
    // json{} return an array
    TimeTransformation tt = json{value.toStdString()}[0].get<TimeTransformation>();

    api_.transform.set_time_transformation(tt);
    parent_->notify();
}

void ImageRenderingPanel::set_time_transformation_size()
{
    api_.transform.update_time_transformation_size(ui_->timeTransformationSizeSpinBox->value());
    parent_->notify();
}

// λ
void ImageRenderingPanel::set_lambda(const double value)
{
    if (api_.transform.set_lambda(static_cast<float>(value) * 1.0e-9f) == ApiCode::OK)
        ui_->BoundaryDoubleSpinBox->setValue(api_.information.get_boundary() * 1000);
}

void ImageRenderingPanel::set_z_distance_slider(int value)
{
    float z_distance = value / 1000.0f;

    api_.transform.set_z_distance(z_distance);

    // Keep consistency between the slider and double box
    const QSignalBlocker blocker(ui_->ZDoubleSpinBox);
    ui_->ZDoubleSpinBox->setValue(value);
}

void ImageRenderingPanel::set_z_distance(const double value)
{
    api_.transform.set_z_distance(static_cast<float>(value) / 1000.0f);

    const QSignalBlocker blocker(ui_->ZSlider);
    ui_->ZSlider->setValue(value);
    ui_->ZDoubleSpinBox->setValue(value);
}

void ImageRenderingPanel::increment_z() { set_z_distance(api_.transform.get_z_distance() * 1000 + z_step_); }

void ImageRenderingPanel::decrement_z() { set_z_distance(api_.transform.get_z_distance() * 1000 - z_step_); }

void ImageRenderingPanel::set_convolution_mode(const bool value)
{
    if (api_.compute.get_is_computation_stopped())
        return;

    if (value)
        api_.global_pp.enable_convolution(api_.global_pp.get_convolution_file_name());
    else
        api_.global_pp.disable_convolution();

    parent_->notify();
}

void ImageRenderingPanel::update_convo_kernel(const QString& value)
{
    std::string v = value.toStdString();
    if (api_.global_pp.enable_convolution(v == UID_CONVOLUTION_TYPE_DEFAULT ? "" : v) == ApiCode::OK)
        parent_->notify();
}

void ImageRenderingPanel::set_divide_convolution(const bool value)
{
    api_.global_pp.set_divide_convolution_enabled(value);
    parent_->notify();
}

double ImageRenderingPanel::get_z_step() { return z_step_; }

} // namespace holovibes::gui
