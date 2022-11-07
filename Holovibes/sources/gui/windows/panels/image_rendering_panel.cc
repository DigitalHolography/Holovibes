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

    ui_->TimeStrideSpinBox->setEnabled(!is_raw);

    ui_->TimeStrideSpinBox->setValue(api::get_time_stride());
    ui_->TimeStrideSpinBox->setSingleStep(api::get_batch_size());
    ui_->TimeStrideSpinBox->setMinimum(api::get_batch_size());

    ui_->BatchSizeSpinBox->setValue(api::get_batch_size());

    ui_->BatchSizeSpinBox->setEnabled(!is_raw && !UserInterfaceDescriptor::instance().is_recording_);

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

    ui_->WaveLengthDoubleSpinBox->setEnabled(!is_raw);
    ui_->WaveLengthDoubleSpinBox->setValue(api::get_lambda() * 1.0e9f);
    ui_->ZDoubleSpinBox->setEnabled(!is_raw);
    ui_->ZDoubleSpinBox->setValue(api::get_z_distance());
    ui_->ZDoubleSpinBox->setSingleStep(z_step_);

    // ViewFilter2D
    ui_->ViewFilter2D->setEnabled(!is_raw);
    ui_->ViewFilter2D->setChecked(api::get_filter2d().enabled);
    ui_->Filter2DView->setEnabled(!is_raw && api::get_filter2d().enabled);
    ui_->Filter2DView->setChecked(!is_raw && api::get_filter2d().enabled);
    ui_->Filter2DN1SpinBox->setEnabled(!is_raw && api::get_filter2d().enabled);
    ui_->Filter2DN1SpinBox->setValue(api::get_filter2d().n1);
    ui_->Filter2DN1SpinBox->setMaximum(ui_->Filter2DN2SpinBox->value() - 1);
    ui_->Filter2DN2SpinBox->setEnabled(!is_raw && api::get_filter2d().enabled);
    ui_->Filter2DN2SpinBox->setValue(api::get_filter2d().n2);

    // Convolution
    ui_->ConvoCheckBox->setEnabled(api::get_compute_mode() == Computation::Hologram);
    ui_->ConvoCheckBox->setChecked(api::get_convolution().enabled);
    ui_->DivideConvoCheckBox->setChecked(api::get_convolution().enabled && api::get_convolution().enabled);
    ui_->KernelQuickSelectComboBox->setCurrentIndex(
        ui_->KernelQuickSelectComboBox->findText(QString::fromStdString(api::get_convolution().type)));
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
    if (api::get_import_type() == ImportTypeEnum::None)
        return;

    if (mode == static_cast<int>(Computation::Raw))
    {
        api::close_windows();
        api::close_critical_compute();

        if (!UserInterfaceDescriptor::instance().is_enabled_camera_)
            return;

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

        const bool res = api::set_holographic_mode(parent_->window_max_size);

        if (res)
        {
            /* ViewFilter2D */
            camera::FrameDescriptor fd = api::get_gpu_input_queue().get_fd();
            ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));

            /* Record Frame Calculation. Only in file mode */
            if (api::get_import_type() == ImportTypeEnum::File)
                ui_->NumberOfFramesSpinBox->setValue(
                    ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                         (float)ui_->TimeStrideSpinBox->value()));

            /* Batch size */
            // The batch size is set with the value present in GUI.
            update_batch_size();

            /* Notify */
            parent_->notify();
        }
    }
}

void ImageRenderingPanel::update_batch_size()
{
    if (api::get_compute_mode() == Computation::Raw || api::get_import_type() == ImportTypeEnum::None)
        return;

    uint batch_size = ui_->BatchSizeSpinBox->value();

    // Need a notify because time transformation stride might change due to change on batch size
    auto notify_callback = [=]() { parent_->notify(); };

    api::update_batch_size(notify_callback, batch_size);
}

void ImageRenderingPanel::update_time_stride()
{
    if (api::get_compute_mode() == Computation::Raw || api::get_import_type() == ImportTypeEnum::None)
        return;

    uint time_stride = ui_->TimeStrideSpinBox->value();

    if (time_stride == api::get_time_stride())
        return;

    auto callback = [=]()
    {
        api::set_time_stride(time_stride);

        // Only in file mode, if batch size change, the record frame number have to change
        // User need.
        if (api::get_import_type() == ImportTypeEnum::File)
            ui_->NumberOfFramesSpinBox->setValue(
                ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                     (float)ui_->TimeStrideSpinBox->value()));
        parent_->notify();
    };

    api::update_time_stride(callback, time_stride);
}

void ImageRenderingPanel::set_filter2d(bool checked)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    api::set_filter2d(checked);

    if (checked)
    {
        // Set the input box related to the filter2d
        const camera::FrameDescriptor& fd = api::get_gpu_input_queue().get_fd();
        ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));
    }
    else
        update_filter2d_view(false);

    parent_->notify();
}

void ImageRenderingPanel::set_filter2d_n1(int n)
{
    api::detail::change_value<Filter2D>()->n1 = n;
    api::set_auto_contrast_all();
}

void ImageRenderingPanel::set_filter2d_n2(int n)
{
    ui_->Filter2DN1SpinBox->setMaximum(n - 1);
    api::detail::change_value<Filter2D>()->n2 = n;
    api::set_auto_contrast_all();
}

void ImageRenderingPanel::update_filter2d_view(bool checked)
{
    if (api::get_compute_mode() == Computation::Raw || api::get_import_type() == ImportTypeEnum::None)
        return;

    api::set_filter2d_view(checked, parent_->auxiliary_window_max_size);
}

void ImageRenderingPanel::set_space_transformation(const QString& value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    SpaceTransformationEnum st;

    try
    {
        // json{} return an array
        st = json{value.toStdString()}[0].get<SpaceTransformationEnum>();
        LOG_DEBUG(main, "value.toStdString() : {}", value.toStdString());
    }
    catch (std::out_of_range& e)
    {
        LOG_ERROR(main, "Catch {}", e.what());
        throw;
    }

    // Prevent useless reload of Holo window
    if (st == api::get_space_transformation())
        return;

    api::set_space_transformation(st);

    // Permit to reset holo window, to apply time transformation change
    set_image_mode(static_cast<int>(Computation::Hologram));
}

void ImageRenderingPanel::set_time_transformation(const QString& value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    // json{} return an array
    TimeTransformationEnum tt = json{value.toStdString()}[0].get<TimeTransformationEnum>();
    LOG_DEBUG(main, "value.toStdString() : {}", value.toStdString());
    // Prevent useless reload of Holo window
    if (api::get_time_transformation() == tt)
        return;

    api::set_time_transformation(tt);

    // Permit to reset holo window, to apply time transformation change
    set_image_mode(static_cast<int>(Computation::Hologram));
}

void ImageRenderingPanel::set_time_transformation_size()
{
    if (api::get_compute_mode() == Computation::Raw || api::get_import_type() == ImportTypeEnum::None)
        return;

    int time_transformation_size = ui_->timeTransformationSizeSpinBox->value();
    time_transformation_size = std::max(1, time_transformation_size);

    if (time_transformation_size == api::get_time_transformation_size())
        return;

    auto callback = [=]()
    {
        api::set_time_transformation_size(time_transformation_size);

        ui_->ViewPanel->set_p_accu();
        // This will not do anything until
        // SliceWindow::changeTexture() isn't coded.
        parent_->notify();
    };

    api::set_time_transformation_size(callback);
}

void ImageRenderingPanel::set_wavelength(const double value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    api::set_lambda(value * 1.0e-9f);
}

void ImageRenderingPanel::set_z(const double value)
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    api::set_z_distance(value);
}

void ImageRenderingPanel::increment_z()
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    set_z(api::get_z_distance() + z_step_);
    ui_->ZDoubleSpinBox->setValue(api::get_z_distance());
}

void ImageRenderingPanel::decrement_z()
{
    if (api::get_compute_mode() == Computation::Raw)
        return;

    set_z(api::get_z_distance() - z_step_);
    ui_->ZDoubleSpinBox->setValue(api::get_z_distance());
}

void ImageRenderingPanel::set_convolution_mode(const bool value)
{
    if (api::get_import_type() == ImportTypeEnum::None)
        return;

    if (value)
        api::enable_convolution(api::get_convolution().type);
    else
        api::disable_convolution();

    parent_->notify();
}

void ImageRenderingPanel::update_convo_kernel(const QString& value)
{
    if (api::get_import_type() == ImportTypeEnum::None)
        return;

    if (!api::get_convolution().enabled)
        return;

    api::get_convolution().type = value.toStdString();
    api::enable_convolution(api::get_convolution().type);

    parent_->notify();
}

void ImageRenderingPanel::set_divide_convolution(const bool value)
{
    if (api::get_import_type() == ImportTypeEnum::None)
        return;

    api::change_convolution()->enabled = value;

    parent_->notify();
}

void ImageRenderingPanel::set_z_step(double value)
{
    z_step_ = value;
    ui_->ZDoubleSpinBox->setSingleStep(value);
}

double ImageRenderingPanel::get_z_step() { return z_step_; }

} // namespace holovibes::gui
