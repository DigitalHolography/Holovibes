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

#include "user_interface.hh"

#include <map>

namespace holovibes::gui
{
ImageRenderingPanel::ImageRenderingPanel(QWidget* parent)
    : Panel(parent)
{
    UserInterface::instance().image_rendering_panel = this;

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

void ImageRenderingPanel::init() { ui_->ZDoubleSpinBox->setSingleStep(z_distance_step_); }

void ImageRenderingPanel::on_notify()
{
    const bool is_raw = api::get_compute_mode() == ComputeModeEnum::Raw;

    ui_->ImageModeComboBox->setCurrentIndex(static_cast<int>(api::get_compute_mode()));

    ui_->TimeStrideSpinBox->setEnabled(!is_raw);
    ui_->TimeStrideSpinBox->setValue(api::get_time_stride());
    ui_->TimeStrideSpinBox->setSingleStep(api::get_batch_size());
    ui_->TimeStrideSpinBox->setMinimum(api::get_batch_size());

    ui_->BatchSizeSpinBox->setEnabled(!is_raw && !api::get_record().is_running);
    ui_->BatchSizeSpinBox->setValue(api::get_batch_size());
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
    ui_->ZDoubleSpinBox->setSingleStep(z_distance_step_);

    // Filter2D
    ui_->Filter2D->setEnabled(!is_raw);
    ui_->Filter2D->setChecked(api::get_filter2d().enabled);
    ui_->Filter2DView->setEnabled(!is_raw && api::get_filter2d().enabled);
    ui_->Filter2DView->setChecked(api::get_filter2d_view_enabled());
    ui_->Filter2DN1SpinBox->setEnabled(!is_raw && api::get_filter2d().enabled);
    ui_->Filter2DN1SpinBox->setValue(api::get_filter2d().inner_radius);
    ui_->Filter2DN1SpinBox->setMaximum(ui_->Filter2DN2SpinBox->value() - 1);
    ui_->Filter2DN2SpinBox->setEnabled(!is_raw && api::get_filter2d().enabled);
    ui_->Filter2DN2SpinBox->setValue(api::get_filter2d().outer_radius);

    // Convolution
    ui_->ConvoCheckBox->setEnabled(api::get_compute_mode() == ComputeModeEnum::Hologram);
    ui_->ConvoCheckBox->setChecked(api::get_convolution().enabled);
    ui_->DivideConvoCheckBox->setChecked(api::get_convolution().enabled && api::get_convolution().divide);
    ui_->KernelQuickSelectComboBox->setCurrentIndex(
        ui_->KernelQuickSelectComboBox->findText(QString::fromStdString(api::get_convolution().type)));
    ui_->KernelQuickSelectComboBox->setEnabled(true);

    FrameDescriptor fd = api::get_import_frame_descriptor();
    ui_->Filter2DN2SpinBox->setMaximum(floor((fmax(fd.width, fd.height) / 2) * M_SQRT2));

    /* Record Frame Calculation. Only in file mode */
    if (api::get_import_type() == ImportTypeEnum::File)
        ui_->NumberOfFramesSpinBox->setValue(
            ceil((ui_->ImportEndIndexSpinBox->value() - ui_->ImportStartIndexSpinBox->value()) /
                 (float)ui_->TimeStrideSpinBox->value()));
}

void ImageRenderingPanel::load_gui(const json& j_us)
{
    z_distance_step_ = json_get_or_default(j_us, z_distance_step_, "gui settings", "z step");
    bool h = json_get_or_default(j_us, isHidden(), "panels", "image rendering hidden");
    ui_->actionImage_rendering->setChecked(!h);
    setHidden(h);
}

void ImageRenderingPanel::save_gui(json& j_us)
{
    j_us["gui settings"]["z step"] = z_distance_step_;
    j_us["panels"]["image rendering hidden"] = isHidden();
}

void ImageRenderingPanel::set_compute_mode(int mode) { api::set_compute_mode(static_cast<ComputeModeEnum>(mode)); }

void ImageRenderingPanel::update_batch_size() { api::set_batch_size(ui_->BatchSizeSpinBox->value()); }

void ImageRenderingPanel::update_time_stride() { api::set_time_stride(ui_->TimeStrideSpinBox->value()); }

void ImageRenderingPanel::set_filter2d(bool checked) { api::change_filter2d()->enabled = checked; }

void ImageRenderingPanel::update_filter2d_n()
{
    ui_->Filter2DN1SpinBox->setMaximum(ui_->Filter2DN2SpinBox->value() - 1);
    api::detail::change_value<Filter2D>()->inner_radius = ui_->Filter2DN1SpinBox->value();
    api::detail::change_value<Filter2D>()->outer_radius = ui_->Filter2DN2SpinBox->value();
}

void ImageRenderingPanel::update_filter2d_view(bool checked)
{
    api::set_filter2d_view_enabled(checked);
    api::get_compute_pipe().get_rendering().request_view_exec_contrast(WindowKind::ViewFilter2D);
}

void ImageRenderingPanel::set_space_transformation(const QString& value)
{
    SpaceTransformationEnum st;
    try
    {
        st = json{value.toStdString()}[0].get<SpaceTransformationEnum>();
    }
    catch (std::out_of_range& e)
    {
        LOG_ERROR("Catch {}", e.what());
        return;
    }

    api::set_space_transformation(st);
}

void ImageRenderingPanel::set_time_transformation(const QString& value)
{
    TimeTransformationEnum tt;
    try
    {
        tt = json{value.toStdString()}[0].get<TimeTransformationEnum>();
    }
    catch (std::out_of_range& e)
    {
        LOG_ERROR("Catch {}", e.what());
        return;
    }

    api::set_time_transformation(tt);
}

void ImageRenderingPanel::set_time_transformation_size()
{
    api::set_time_transformation_size(ui_->timeTransformationSizeSpinBox->value());
}

void ImageRenderingPanel::update_wavelength() { api::set_lambda(ui_->WaveLengthDoubleSpinBox->value() * 1.0e-9f); }

void ImageRenderingPanel::update_z_distance() { api::set_z_distance(ui_->ZDoubleSpinBox->value()); }

void ImageRenderingPanel::increment_z() { api::set_z_distance(api::get_z_distance() + z_distance_step_); }

void ImageRenderingPanel::decrement_z() { api::set_z_distance(api::get_z_distance() - z_distance_step_); }

void ImageRenderingPanel::set_convolution_mode(const bool value) { api::change_convolution()->enabled = value; }

void ImageRenderingPanel::update_convo_kernel(const QString& value)
{
    api::change_convolution()->type = value.toStdString();
}

void ImageRenderingPanel::set_divide_convolution(const bool value) { api::change_convolution()->divide = value; }

void ImageRenderingPanel::set_z_distance_step(double value)
{
    z_distance_step_ = value;
    ui_->ZDoubleSpinBox->setSingleStep(value);
}

} // namespace holovibes::gui
