#include "compute_settings_struct.hh"
#include "API.hh"

namespace holovibes
{

void Windows::Update()
{
    this->xy = GSH::instance().get_xy();
    this->xy = GSH::instance().get_xy();
    this->xz = GSH::instance().get_xz();
    this->filter2d = GSH::instance().get_filter2d();
}

void Reticle::Update()
{
    this->display_enabled = GSH::instance().get_reticle_display_enabled();
    this->reticle_scale = GSH::instance().get_reticle_scale();
}

void Views::Update()
{
    this->img_type = GSH::instance().get_img_type();
    this->fft_shift = GSH::instance().get_fft_shift_enabled();
    this->x = GSH::instance().get_x();
    this->y = GSH::instance().get_y();
    this->p = GSH::instance().get_p();
    this->q = GSH::instance().get_q();
    this->window.Update();
    this->renorm = GSH::instance().get_renorm_enabled();
    this->reticle.Update();
}

void Rendering::Convolution::Update()
{
    this->enabled = GSH::instance().get_convolution_enabled();
    this->type = UserInterfaceDescriptor::instance().convo_name;
    this->divide = GSH::instance().get_divide_convolution_enabled();
}

void Rendering::Filter2D::Update()
{
    this->enabled = GSH::instance().get_filter2d_enabled();
    this->n1 = GSH::instance().get_filter2d_n1();
    this->n2 = GSH::instance().get_filter2d_n2();
}

void Rendering::Update()
{
    this->image_mode = GSH::instance().get_compute_mode();
    this->batch_size = GSH::instance().get_batch_size();
    this->time_transformation_stride = GSH::instance().get_time_stride();
    this->filter2d.Update();
    this->space_transformation = GSH::instance().get_space_transformation();
    this->time_transformation = GSH::instance().get_time_transformation();
    this->time_transformation_size = GSH::instance().get_time_transformation_size();
    this->lambda = GSH::instance().get_lambda();
    this->z_distance = GSH::instance().get_z_distance();
    this->convolution.Update();
}

void AdvancedSettings::BufferSizes::Update()
{
    this->input = GSH::instance().get_file_buffer_size();
    this->file = GSH::instance().get_input_buffer_size();
    this->record = GSH::instance().get_output_buffer_size();
    this->output = GSH::instance().get_record_buffer_size();
    this->time_transformation_cuts = GSH::instance().get_time_transformation_cuts_output_buffer_size();
}

void AdvancedSettings::Filter2DSmooth::Update()
{
    this->low = GSH::instance().get_filter2d_smooth_low();
    this->high = GSH::instance().get_filter2d_smooth_high();
}

void AdvancedSettings::ContrastThreshold::Update()
{
    this->lower = GSH::instance().get_contrast_lower_threshold();
    this->upper = GSH::instance().get_contrast_upper_threshold();
    this->cuts_p_offset = GSH::instance().get_cuts_contrast_p_offset();
}

void AdvancedSettings::Update()
{
    this->buffer_size.Update();
    this->filter2d_smooth.Update();
    this->contrast.Update();
    this->renorm_constant = GSH::instance().get_renorm_constant();
}

void Composite::Update()
{
    this->mode = GSH::instance().get_composite_kind();
    this->composite_auto_weights = GSH::instance().get_composite_auto_weights();
    this->rgb = GSH::instance().get_rgb();
    this->hsv = GSH::instance().get_hsv();
}

void ComputeSettings::Update()
{
    this->image_rendering.Update();
    this->view.Update();
    this->composite.Update();
    this->advanced.Update();
}

} // namespace holovibes