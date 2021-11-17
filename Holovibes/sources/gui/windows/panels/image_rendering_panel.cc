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
    const bool is_raw = api::is_raw_mode();

    ui_->TimeTransformationStrideSpinBox->setEnabled(!is_raw);

    ui_->TimeTransformationStrideSpinBox->setValue(api::get_time_transformation_stride());
    ui_->TimeTransformationStrideSpinBox->setSingleStep(api::get_batch_size());
    ui_->TimeTransformationStrideSpinBox->setMinimum(api::get_batch_size());

    ui_->BatchSizeSpinBox->setEnabled(!is_raw && !UserInterfaceDescriptor::instance().is_recording_);

    api::check_batch_size_limit();
    ui_->BatchSizeSpinBox->setValue(api::get_batch_size());
    ui_->BatchSizeSpinBox->setMaximum(api::get_input_buffer_size());

    ui_->SpaceTransformationComboBox->setEnabled(!is_raw && !api::get_3d_cuts_view_enabled());
    ui_->SpaceTransformationComboBox->setCurrentIndex(static_cast<int>(api::get_space_transformation()));
    ui_->TimeTransformationComboBox->setEnabled(!is_raw);
    ui_->TimeTransformationComboBox->setCurrentIndex(static_cast<int>(api::get_time_transformation()));

    // Changing time_transformation_size with time transformation cuts is
    // supported by the pipe, but some modifications have to be done in
    // SliceWindow, OpenGl buffers.
    ui_->timeTransformationSizeSpinBox->setEnabled(!is_raw && !api::get_3d_cuts_view_enabled());
    ui_->timeTransformationSizeSpinBox->setValue(api::get_time_transformation_size());

    ui_->WaveLengthDoubleSpinBox->setEnabled(!is_raw);
    ui_->WaveLengthDoubleSpinBox->setValue(api::get_lambda() * 1.0e9f);
    ui_->ZDoubleSpinBox->setEnabled(!is_raw);
    ui_->ZDoubleSpinBox->setValue(api::get_zdistance());
    ui_->ZDoubleSpinBox->setSingleStep(z_step_);
    ui_->BoundaryLineEdit->setText(QString::number(api::get_boundary()));

    // Filter2D
    ui_->Filter2D->setEnabled(!is_raw);
    ui_->Filter2D->setChecked(!is_raw && api::get_filter2d_enabled());
    ui_->Filter2DView->setEnabled(!is_raw && api::get_filter2d_enabled());
    ui_->Filter2DView->setChecked(!is_raw && api::get_filter2d_view_enabled());
    ui_->Filter2DN1SpinBox->setEnabled(!is_raw && api::get_filter2d_enabled());
    ui_->Filter2DN1SpinBox->setValue(api::get_filter2d_n1());
    ui_->Filter2DN1SpinBox->setMaximum(ui_->Filter2DN2SpinBox->value() - 1);
    ui_->Filter2DN2SpinBox->setEnabled(!is_raw && api::get_filter2d_enabled());
    ui_->Filter2DN2SpinBox->setValue(api::get_filter2d_n2());

    // Convolution
    ui_->ConvoCheckBox->setEnabled(api::get_compute_mode() == Computation::Hologram);
    ui_->ConvoCheckBox->setChecked(api::get_convolution_enabled());
    ui_->DivideConvoCheckBox->setChecked(api::get_convolution_enabled() && api::get_divide_convolution_enabled());
}

void ImageRenderingPanel::load_gui(const boost::property_tree::ptree& ptree)
{
    // Step
    z_step_ = ptree.get<double>("gui_settings.image_rendering_z_step", z_step_);
    bool h = ptree.get<bool>("window.image_rendering_hidden", isHidden());
    ui_->actionImage_rendering->setChecked(!h);
    setHidden(h);
}

void ImageRenderingPanel::save_gui(boost::property_tree::ptree& ptree)
{
    ptree.put<double>("gui_settings.image_rendering_z_step", z_step_);
    ptree.put<bool>("window.image_rendering_hidden", isHidden());
}

void ImageRenderingPanel::set_image_mode(int mode)
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    if (mode == static_cast<int>(Computation::Raw)) // Raw
    {
        api::close_windows();
        parent_->notify();
        parent_->layout_toggled();

        api::close_critical_compute();
        parent_->notify();

        if (!UserInterfaceDescriptor::instance().is_enabled_camera_)
            return;

        api::set_raw_mode(*parent_, parent_->window_max_size);

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

        camera::FrameDescriptor fd;
        const bool res = api::set_holographic_mode(*parent_, parent_->window_max_size, fd);

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
}

void ImageRenderingPanel::set_raw_mode() {}

void ImageRenderingPanel::set_holographic_mode() {}

void ImageRenderingPanel::update_batch_size()
{
    if (api::is_raw_mode() || UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
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
    if (api::is_raw_mode() || UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
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

    api::set_filter2d(checked);

    if (checked)
    {
        // Set the input box related to the filter2d
        const camera::FrameDescriptor& fd = api::get_fd();
        ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));
    }
    else
        update_filter2d_view(false);

    parent_->notify();
}

void ImageRenderingPanel::set_filter2d_n1(int n) { api::set_filter2d_n1(n); }

void ImageRenderingPanel::set_filter2d_n2(int n) { api::set_filter2d_n2(n); }

void ImageRenderingPanel::update_filter2d_view(bool checked)
{
    if (UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    // To move in api function
    if (api::is_raw_mode())
        return;

    api::set_filter2d_view(checked, parent_->auxiliary_window_max_size);

    if (!checked && UserInterfaceDescriptor::instance().filter2d_window)
    {
        // Remove the on triggered event
        disconnect(UserInterfaceDescriptor::instance().filter2d_window.get(),
                   SIGNAL(destroyed()),
                   this,
                   SLOT(update_filter2d_view(false)));
    }

    parent_->notify();
}

void ImageRenderingPanel::set_space_transformation(const QString& value)
{
    if (api::is_raw_mode())
        return;

    // String are set according to value in the appropriate ComboBox
    static std::map<std::string, SpaceTransformation> space_transformation_dictionary = {
        {"None", SpaceTransformation::NONE},
        {"1FFT", SpaceTransformation::FFT1},
        {"2FFT", SpaceTransformation::FFT2},
    };

    auto st = space_transformation_dictionary.at(value.toStdString());
    // Prevent useless reload of Holo window
    if (st == api::get_space_transformation())
        return;

    api::set_space_transformation(st);

    // Permit to reset holo window, to apply time transformation change
    set_image_mode(static_cast<int>(Computation::Hologram));
}

void ImageRenderingPanel::set_time_transformation(const QString& value)
{
    if (api::is_raw_mode())
        return;

    // String are set according to value in the appropriate ComboBox
    static std::map<std::string, TimeTransformation> time_transformation_dictionary = {
        {"None", TimeTransformation::NONE},
        {"PCA", TimeTransformation::PCA},
        {"SSA_STFT", TimeTransformation::SSA_STFT},
        {"STFT", TimeTransformation::STFT},
    };

    TimeTransformation tt = time_transformation_dictionary.at(value.toStdString());
    // Prevent useless reload of Holo window
    if (api::get_time_transformation() == tt)
        return;

    api::set_time_transformation(time_transformation_dictionary.at(value.toStdString()));

    // Permit to reset holo window, to apply time transformation change
    set_image_mode(static_cast<int>(Computation::Hologram));
}

void ImageRenderingPanel::set_time_transformation_size()
{
    if (api::is_raw_mode() || UserInterfaceDescriptor::instance().import_type_ == ImportType::None)
        return;

    int time_transformation_size = ui_->timeTransformationSizeSpinBox->value();
    time_transformation_size = std::max(1, time_transformation_size);

    if (time_transformation_size == api::get_time_transformation_size())
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

    api::set_wavelength(value * 1.0e-9f);
}

void ImageRenderingPanel::set_z(const double value)
{
    if (api::is_raw_mode())
        return;

    api::set_z_distance(value);
}

void ImageRenderingPanel::increment_z()
{
    set_z(api::get_zdistance() + z_step_);
    ui_->ZDoubleSpinBox->setValue(api::get_zdistance());
}

void ImageRenderingPanel::decrement_z()
{
    set_z(api::get_zdistance() - z_step_);
    ui_->ZDoubleSpinBox->setValue(api::get_zdistance());
}

void ImageRenderingPanel::set_convolution_mode(const bool value)
{
    if (value)
    {
        std::string str = ui_->KernelQuickSelectComboBox->currentText().toStdString();
        api::set_convolution_mode(str);
    }
    else
        api::unset_convolution_mode();

    parent_->notify();
}

void ImageRenderingPanel::update_convo_kernel(const QString& value)
{
    if (!api::get_convolution_enabled())
        return;

    api::update_convo_kernel(value.toStdString());

    parent_->notify();
}

void ImageRenderingPanel::set_divide_convolution(const bool value)
{
    api::set_divide_convolution(value);

    parent_->notify();
}
} // namespace holovibes::gui
