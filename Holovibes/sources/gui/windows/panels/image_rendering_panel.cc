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

#include <map>

namespace api = ::holovibes::api;

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
    const bool is_raw = api::get_compute_mode() == Computation::Raw;

    ui_->ImageModeComboBox->setCurrentIndex(static_cast<int>(api::get_compute_mode()));
    ui_->ImageModeComboBox->setEnabled((api::get_input_queue_location() == holovibes::Device::GPU));

    ui_->TimeStrideSpinBox->setEnabled(!is_raw);

    ui_->TimeStrideSpinBox->setValue(api::get_time_stride());
    ui_->TimeStrideSpinBox->setSingleStep(api::get_batch_size());
    ui_->TimeStrideSpinBox->setMinimum(api::get_batch_size());

    ui_->BatchSizeSpinBox->setValue(api::get_batch_size());

    ui_->BatchSizeSpinBox->setEnabled(!UserInterfaceDescriptor::instance().is_recording_);

    ui_->BatchSizeSpinBox->setMaximum(api::get_input_buffer_size());

    ui_->SpaceTransformationComboBox->setEnabled(!is_raw);
    ui_->SpaceTransformationComboBox->setCurrentIndex(static_cast<int>(api::get_space_transformation()));
    ui_->TimeTransformationComboBox->setEnabled(!is_raw);
    ui_->TimeTransformationComboBox->setCurrentIndex(static_cast<int>(api::get_time_transformation()));

    // Changing time_transformation_size with time transformation cuts is
    // supported by the pipe, but some modifications have to be done in
    // SliceWindow, OpenGl buffers.
    ui_->timeTransformationSizeSpinBox->setEnabled(!is_raw && !api::get_cuts_view_enabled());
    ui_->timeTransformationSizeSpinBox->setValue(api::get_time_transformation_size());

    // Z (focus)
    ui_->LambdaSpinBox->setEnabled(!is_raw);
    ui_->LambdaSpinBox->setValue(api::get_lambda() * 1.0e9f);
    ui_->ZDoubleSpinBox->setEnabled(!is_raw);
    ui_->ZDoubleSpinBox->setValue(api::get_z_distance() * 1000);
    ui_->ZDoubleSpinBox->setSingleStep(z_step_);
    ui_->BoundaryDoubleSpinBox->setValue(api::get_boundary() * 1000);

    // Filter2D
    bool filter2D_enabled = !is_raw && api::get_filter2d_enabled();
    ui_->Filter2D->setEnabled(!is_raw);
    ui_->Filter2D->setChecked(filter2D_enabled);

    ui_->Filter2DView->setVisible(filter2D_enabled);
    ui_->Filter2DView->setChecked(!is_raw && api::get_filter2d_view_enabled());
    ui_->Filter2DN1SpinBox->setVisible(filter2D_enabled);
    ui_->Filter2DN1SpinBox->setValue(api::get_filter2d_n1());

    ui_->Filter2DN2SpinBox->setVisible(filter2D_enabled);
    ui_->Filter2DN2SpinBox->setValue(api::get_filter2d_n2());
    ui_->Filter2DN1SpinBox->setMaximum(ui_->Filter2DN2SpinBox->value() - 1);

    // Filter
    ui_->InputFilterLabel->setVisible(filter2D_enabled);
    ui_->InputFilterQuickSelectComboBox->setVisible(filter2D_enabled);
    if (!api::get_filter_enabled())
    {
        ui_->InputFilterQuickSelectComboBox->setCurrentIndex(
            ui_->InputFilterQuickSelectComboBox->findText(UID_FILTER_TYPE_DEFAULT));
    }
    else
    {
        ui_->InputFilterQuickSelectComboBox->setCurrentIndex(ui_->InputFilterQuickSelectComboBox->findText(
            QString::fromStdString(UserInterfaceDescriptor::instance().filter_name)));
    }

    // Convolution
    ui_->ConvoCheckBox->setVisible(api::get_compute_mode() == Computation::Hologram);
    ui_->ConvoCheckBox->setChecked(api::get_convolution_enabled());

    ui_->DivideConvoCheckBox->setVisible(api::get_convolution_enabled());
    ui_->DivideConvoCheckBox->setChecked(api::get_divide_convolution_enabled());
    ui_->KernelQuickSelectComboBox->setVisible(api::get_convolution_enabled());
    ui_->KernelQuickSelectComboBox->setCurrentIndex(ui_->KernelQuickSelectComboBox->findText(
        QString::fromStdString(UserInterfaceDescriptor::instance().convo_name)));
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

void ImageRenderingPanel::set_image_mode(int mode)
{
    if (api::get_import_type() == ImportType::None)
        return;

    if (mode == static_cast<int>(Computation::Raw))
    {
        api::close_windows();
        api::close_critical_compute();
        api::set_raw_mode(parent_->window_max_size);

        parent_->notify();
        parent_->layout_toggled();
    }
    else if (mode == static_cast<int>(Computation::Hologram))
    {
        // That function is used to reallocate the buffers since the Square
        // input mode could have changed
        /* Close windows & destory thread compute */
        api::close_windows();
        api::close_critical_compute();

        api::change_window(static_cast<int>(WindowKind::XYview));

        api::set_holographic_mode(parent_->window_max_size);

        /* Filter2D */
        camera::FrameDescriptor fd = api::get_fd();
        ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));

        /* Record Frame Calculation. Only in file mode */
        if (api::get_import_type() == ImportType::File)
            ui_->NumberOfFramesSpinBox->setValue(
                ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                     (float)ui_->TimeStrideSpinBox->value()));

        /* Batch size */
        // The batch size is set with the value present in GUI.
        // update_batch_size();

        /* Notify */
        parent_->notify();
    }
}

void ImageRenderingPanel::update_batch_size()
{
    api::update_batch_size(ui_->BatchSizeSpinBox->value());
    parent_->notify();
}

void ImageRenderingPanel::update_time_stride()
{
    api::update_time_stride(ui_->TimeStrideSpinBox->value());

    if (api::get_import_type() == ImportType::File)
        ui_->NumberOfFramesSpinBox->setValue(
            ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                 (float)ui_->TimeStrideSpinBox->value()));
    parent_->notify();
}

void ImageRenderingPanel::set_filter2d(bool checked)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    api::set_filter2d(checked);

    if (checked)
    {
        // Set the input box related to the filter2d
        const camera::FrameDescriptor& fd = api::get_fd();
        const int size_max = floor((fmax(fd.width, fd.height) / 2) * M_SQRT2);
        ui_->Filter2DN2SpinBox->setMaximum(size_max);
        // sets the filter_2d_n2 so the frame fits in the lens diameter by default
        api::set_filter2d_n2(size_max);
        ui_->Filter2DN2SpinBox->setValue(size_max);
    }
    else
        update_filter2d_view(false);

    parent_->notify();
}

void ImageRenderingPanel::set_filter2d_n1(int n) { api::set_filter2d_n1(n); }

void ImageRenderingPanel::set_filter2d_n2(int n)
{
    ui_->Filter2DN1SpinBox->setMaximum(n - 1);
    api::set_filter2d_n2(n);
}

void ImageRenderingPanel::update_input_filter(const QString& value)
{
    LOG_FUNC();

    if (value.toStdString() != UserInterfaceDescriptor::instance().filter_name)
    {
        api::enable_filter(value.toStdString());
        parent_->notify();
    }
}

void ImageRenderingPanel::update_filter2d_view(bool checked)
{
    api::set_filter2d_view(checked, parent_->auxiliary_window_max_size);
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

    api::set_space_transformation(st);
    parent_->notify();
}

void ImageRenderingPanel::set_time_transformation(const QString& value)
{
    // json{} return an array
    TimeTransformation tt = json{value.toStdString()}[0].get<TimeTransformation>();

    api::set_time_transformation(tt);
    parent_->notify();
}

void ImageRenderingPanel::set_time_transformation_size()
{
    api::update_time_transformation_size(ui_->timeTransformationSizeSpinBox->value());
    parent_->notify();
}

// λ
void ImageRenderingPanel::set_lambda(const double value)
{
    api::set_lambda(static_cast<float>(value) * 1.0e-9f);
    ui_->BoundaryDoubleSpinBox->setValue(api::get_boundary() * 1000);
}

void ImageRenderingPanel::set_z_distance_slider(int value)
{
    float z_distance = value / 1000.0f;

    api::set_z_distance(z_distance);

    // Keep consistency between the slider and double box
    const QSignalBlocker blocker(ui_->ZDoubleSpinBox);
    ui_->ZDoubleSpinBox->setValue(value);
}

void ImageRenderingPanel::set_z_distance(const double value)
{
    api::set_z_distance(static_cast<float>(value) / 1000.0f);

    const QSignalBlocker blocker(ui_->ZSlider);
    ui_->ZSlider->setValue(value);
}

void ImageRenderingPanel::increment_z() { set_z_distance(api::get_z_distance() + z_step_); }

void ImageRenderingPanel::decrement_z() { set_z_distance(api::get_z_distance() - z_step_); }

void ImageRenderingPanel::set_convolution_mode(const bool value)
{
    if (api::get_import_type() == ImportType::None)
        return;

    if (value)
        api::enable_convolution(UserInterfaceDescriptor::instance().convo_name);
    else
        api::disable_convolution();

    parent_->notify();
}

void ImageRenderingPanel::update_convo_kernel(const QString& value)
{
    UserInterfaceDescriptor::instance().convo_name = value.toStdString();
    api::enable_convolution(UserInterfaceDescriptor::instance().convo_name);
    parent_->notify();
}

void ImageRenderingPanel::set_divide_convolution(const bool value)
{
    api::set_divide_convolution(value);
    parent_->notify();
}

double ImageRenderingPanel::get_z_step() { return z_step_; }

} // namespace holovibes::gui
