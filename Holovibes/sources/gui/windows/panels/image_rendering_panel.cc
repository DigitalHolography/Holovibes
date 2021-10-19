#include <filesystem>

#include "image_rendering_panel.hh"
#include "MainWindow.hh"
#include "logger.hh"
#include "tools.hh"
#include "frame_desc.hh"
#include "API.hh"

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
    const bool is_raw = api::is_raw_mode();

    ui_->TimeTransformationStrideSpinBox->setEnabled(!is_raw);

    const uint input_queue_capacity = global::global_config.input_queue_max_size;

    ui_->TimeTransformationStrideSpinBox->setValue(api::get_cd().time_transformation_stride);
    ui_->TimeTransformationStrideSpinBox->setSingleStep(api::get_cd().batch_size);
    ui_->TimeTransformationStrideSpinBox->setMinimum(api::get_cd().batch_size);

    ui_->BatchSizeSpinBox->setEnabled(!is_raw && !UserInterfaceDescriptor::instance().is_recording_);

    api::get_cd().check_batch_size_limit(input_queue_capacity);
    ui_->BatchSizeSpinBox->setValue(api::get_cd().batch_size);
    ui_->BatchSizeSpinBox->setMaximum(input_queue_capacity);

    ui_->SpaceTransformationComboBox->setEnabled(!is_raw && !api::get_cd().time_transformation_cuts_enabled);
    ui_->SpaceTransformationComboBox->setCurrentIndex(static_cast<int>(api::get_cd().space_transformation.load()));
    ui_->TimeTransformationComboBox->setEnabled(!is_raw);
    ui_->TimeTransformationComboBox->setCurrentIndex(static_cast<int>(api::get_cd().time_transformation.load()));

    // Changing time_transformation_size with time transformation cuts is
    // supported by the pipe, but some modifications have to be done in
    // SliceWindow, OpenGl buffers.
    ui_->timeTransformationSizeSpinBox->setEnabled(!is_raw && !api::get_cd().time_transformation_cuts_enabled);
    ui_->timeTransformationSizeSpinBox->setValue(api::get_cd().time_transformation_size);

    ui_->WaveLengthDoubleSpinBox->setEnabled(!is_raw);
    ui_->WaveLengthDoubleSpinBox->setValue(api::get_cd().lambda * 1.0e9f);
    ui_->ZDoubleSpinBox->setEnabled(!is_raw);
    ui_->ZDoubleSpinBox->setValue(api::get_cd().zdistance);
    ui_->BoundaryLineEdit->setText(QString::number(api::get_boundary()));

    // Filter2D
    ui_->Filter2D->setEnabled(!is_raw);
    ui_->Filter2D->setChecked(!is_raw && api::get_cd().filter2d_enabled);
    ui_->Filter2DView->setEnabled(!is_raw && api::get_cd().filter2d_enabled);
    ui_->Filter2DView->setChecked(!is_raw && api::get_cd().filter2d_view_enabled);
    ui_->Filter2DN1SpinBox->setEnabled(!is_raw && api::get_cd().filter2d_enabled);
    ui_->Filter2DN1SpinBox->setValue(api::get_cd().filter2d_n1);
    ui_->Filter2DN1SpinBox->setMaximum(ui_->Filter2DN2SpinBox->value() - 1);
    ui_->Filter2DN2SpinBox->setEnabled(!is_raw && api::get_cd().filter2d_enabled);
    ui_->Filter2DN2SpinBox->setValue(api::get_cd().filter2d_n2);

    // Convolution
    ui_->ConvoCheckBox->setEnabled(api::get_cd().compute_mode == Computation::Hologram);
    ui_->ConvoCheckBox->setChecked(api::get_cd().convolution_enabled);
    ui_->DivideConvoCheckBox->setChecked(api::get_cd().convolution_enabled && api::get_cd().divide_convolution_enabled);
}

void ImageRenderingPanel::load_ini(const boost::property_tree::ptree& ptree)
{
    ui_->ImageRenderingPanel->setChecked(!ptree.get<bool>("image_rendering.hidden", isHidden()));

    const float z_step = ptree.get<float>("image_rendering.z_step", z_step_);
    if (z_step > 0.0f)
        ui_->ZDoubleSpinBox->setSingleStep(z_step);
}

void ImageRenderingPanel::save_ini(boost::property_tree::ptree& ptree)
{
    ptree.put<bool>("image_rendering.hidden", isHidden());
    ptree.put<double>("image_rendering.z_step", z_step_);
}

void ImageRenderingPanel::set_image_mode(QString mode)
{
    if (mode != nullptr)
    {
        // Call comes from ui
        if (ui_->ImageModeComboBox->currentIndex() == 0)
            set_raw_mode();
        else
            set_holographic_mode();
    }
    else if (api::get_cd().compute_mode == Computation::Raw)
        set_raw_mode();
    else if (api::get_cd().compute_mode == Computation::Hologram)
        set_holographic_mode();
}

void ImageRenderingPanel::set_raw_mode()
{
    api::close_windows();
    parent_->notify();
    parent_->layout_toggled();

    api::close_critical_compute();
    parent_->notify();

    if (!UserInterfaceDescriptor::instance().is_enabled_camera_)
        return;

    api::set_raw_mode(*parent_);

    parent_->notify();
    parent_->layout_toggled();
}

void ImageRenderingPanel::set_holographic_mode()
{

    // That function is used to reallocate the buffers since the Square
    // input mode could have changed
    /* Close windows & destory thread compute */
    api::close_windows();
    api::close_critical_compute();

    camera::FrameDescriptor fd;
    const bool res = api::set_holographic_mode(*parent_, fd);

    if (res)
    {
        /* Filter2D */
        ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));

        /* Record Frame Calculation */
        ui_->NumberOfFramesSpinBox->setValue(
            ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                 (float)ui_->TimeTransformationStrideSpinBox->value()));

        /* Notify */
        parent_->notify();
    }
}

void ImageRenderingPanel::set_computation_mode()
{
    if (ui_->ImageModeComboBox->currentIndex() == 0)
    {
        api::set_computation_mode(Computation::Raw);
    }
    else if (ui_->ImageModeComboBox->currentIndex() == 1)
    {
        api::set_computation_mode(Computation::Hologram);
    }
}

void ImageRenderingPanel::update_batch_size()
{
    if (api::is_raw_mode())
        return;

    uint batch_size = ui_->BatchSizeSpinBox->value();

    if (batch_size == api::get_batch_size())
        return;

    auto callback = [=]() {
        api::set_batch_size(batch_size);
        api::adapt_time_transformation_stride_to_batch_size();
        Holovibes::instance().get_compute_pipe()->request_update_batch_size();
        parent_->notify();
    };

    api::update_batch_size(callback, batch_size);
}

void ImageRenderingPanel::update_time_transformation_stride()
{
    if (api::is_raw_mode())
        return;

    uint time_transformation_stride = ui_->TimeTransformationStrideSpinBox->value();

    if (time_transformation_stride == api::get_time_transformation_stride())
        return;

    auto callback = [=]() {
        api::set_time_transformation_stride(time_transformation_stride);
        api::adapt_time_transformation_stride_to_batch_size();
        Holovibes::instance().get_compute_pipe()->request_update_time_transformation_stride();
        ui_->NumberOfFramesSpinBox->setValue(
            ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                 (float)ui_->TimeTransformationStrideSpinBox->value()));
        parent_->notify();
    };

    api::update_time_transformation_stride(callback, time_transformation_stride);
}

void ImageRenderingPanel::set_filter2d(bool checked)
{
    if (api::is_raw_mode())
        return;

    if (checked)
    {
        api::set_filter2d();

        // Set the input box related to the filter2d
        const camera::FrameDescriptor& fd = api::get_fd();
        ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));
        set_filter2d_n2(ui_->Filter2DN2SpinBox->value());
        set_filter2d_n1(ui_->Filter2DN1SpinBox->value());
    }
    else
    {
        cancel_filter2d();
    }

    parent_->notify();
}

void ImageRenderingPanel::cancel_filter2d()
{
    if (api::is_raw_mode())
        return;

    api::cancel_filter2d();

    if (api::get_filter2d_view_enabled())
        update_filter2d_view(false);

    parent_->notify();
}

void ImageRenderingPanel::set_filter2d_n1(int n)
{
    api::set_filter2d_n1(n);
    api::set_auto_contrast_all();
}

void ImageRenderingPanel::set_filter2d_n2(int n)
{
    api::set_filter2d_n2(n);
    api::set_auto_contrast_all();
}

void ImageRenderingPanel::update_filter2d_view(bool checked)
{
    if (api::is_raw_mode())
        return;

    if (checked)
    {
        api::set_filter2d_view(*parent_);
    }
    else
    {
        disable_filter2d_view();
    }

    parent_->notify();
}

void ImageRenderingPanel::disable_filter2d_view()
{

    if (UserInterfaceDescriptor::instance().filter2d_window)
    {
        // Remove the on triggered event
        disconnect(UserInterfaceDescriptor::instance().filter2d_window.get(),
                   SIGNAL(destroyed()),
                   this,
                   SLOT(disable_filter2d_view()));
    }

    api::disable_filter2d_view();

    // Change the focused window
    parent_->change_window();

    parent_->notify();
}

void ImageRenderingPanel::set_space_transformation(const QString& value)
{
    if (api::is_raw_mode())
        return;

    api::set_space_transformation(value.toStdString());

    set_holographic_mode();
}

void ImageRenderingPanel::set_time_transformation(const QString& value)
{
    if (api::is_raw_mode())
        return;

    api::set_time_transformation(value.toStdString());

    set_holographic_mode();
}

void ImageRenderingPanel::set_time_transformation_size()
{
    if (api::is_raw_mode())
        return;

    int time_transformation_size = ui_->timeTransformationSizeSpinBox->value();
    time_transformation_size = std::max(1, time_transformation_size);

    if (time_transformation_size == api::get_cd().time_transformation_size)
        return;

    auto callback = [=]() {
        api::set_time_transformation_size(time_transformation_size);
        api::get_compute_pipe()->request_update_time_transformation_size();
        ui_->ViewPanel->set_p_accu();
        // This will not do anything until
        // SliceWindow::changeTexture() isn't coded.
    };

    api::set_time_transformation_size(callback);

    parent_->notify();
}

void ImageRenderingPanel::set_wavelength(const double value)
{
    if (api::is_raw_mode())
        return;

    api::set_wavelength(value);
}

void ImageRenderingPanel::set_z(const double value)
{
    if (api::is_raw_mode())
        return;

    api::set_z(value);
}

void ImageRenderingPanel::set_z_step(const double value)
{
    z_step_ = value;
    ui_->ZDoubleSpinBox->setSingleStep(value);
}

void ImageRenderingPanel::increment_z()
{
    if (api::is_raw_mode())
        return;

    set_z(api::get_cd().zdistance + z_step_);
    ui_->ZDoubleSpinBox->setValue(api::get_cd().zdistance);
}

void ImageRenderingPanel::decrement_z()
{
    if (api::is_raw_mode())
        return;

    set_z(api::get_cd().zdistance - z_step_);
    ui_->ZDoubleSpinBox->setValue(api::get_cd().zdistance);
}

void ImageRenderingPanel::set_convolution_mode(const bool value)
{
    if (value)
    {
        std::string str = ui_->KernelQuickSelectComboBox->currentText().toStdString();

        api::set_convolution_mode(str);
    }
    else
    {
        api::unset_convolution_mode();
    }

    parent_->notify();
}

void ImageRenderingPanel::update_convo_kernel(const QString& value)
{
    if (!api::get_convolution_enabled())
        return;

    api::update_convo_kernel(value.toStdString());

    parent_->notify();
}

void ImageRenderingPanel::set_divide_convolution_mode(const bool value)
{
    api::set_divide_convolution_mode(value);
    
    parent_->notify();
}
} // namespace holovibes::gui
