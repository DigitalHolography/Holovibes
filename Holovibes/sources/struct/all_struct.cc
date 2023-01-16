#include <iomanip>
#include <filesystem>
#include <fstream>

#include "compute_settings_struct.hh"
#include "API.hh"
#include "all_struct.hh"

namespace holovibes
{

void Windows::Update()
{
    this->xy = GSH::instance().get_xy();
    this->yz = GSH::instance().get_yz();
    this->xz = GSH::instance().get_xz();
    this->filter2d = GSH::instance().get_filter2d();
}

void Reticle::Update()
{
    this->display_enabled = GSH::instance().get_reticle_display_enabled();
    this->scale = GSH::instance().get_reticle_scale();
}

void Views::Update()
{
    this->image_type = GSH::instance().get_img_type();
    this->fft_shift = GSH::instance().get_fft_shift_enabled();
    this->x = GSH::instance().get_x();
    this->y = GSH::instance().get_y();
    this->z = GSH::instance().get_p();
    this->z2 = GSH::instance().get_q();
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
    this->inner_radius = GSH::instance().get_filter2d_n1();
    this->outer_radius = GSH::instance().get_filter2d_n2();
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
    this->propagation_distance = GSH::instance().get_z_distance();
    this->convolution.Update();
}

void AdvancedSettings::BufferSizes::Update()
{
    this->file = GSH::instance().get_file_buffer_size();
    this->input = GSH::instance().get_input_buffer_size();
    this->output = GSH::instance().get_output_buffer_size();
    this->record = GSH::instance().get_record_buffer_size();
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
    this->frame_index_offset = GSH::instance().get_cuts_contrast_p_offset();
}

void AdvancedSettings::Update()
{
    this->buffer_size.Update();
    this->filter2d_smooth.Update();
    this->contrast.Update();
    this->renorm_constant = GSH::instance().get_renorm_constant();
    this->raw_bitshift = GSH::instance().get_raw_bitshift();
}

void Composite::Update()
{
    this->mode = GSH::instance().get_composite_kind();
    this->auto_weight = GSH::instance().get_composite_auto_weights();
    this->rgb = GSH::instance().get_rgb();
    this->hsv = GSH::instance().get_hsv();
}

void ComputeSettings::Update()
{
    this->image_rendering.Update();
    this->view.Update();
    this->color_composite_image.Update();
    this->advanced.Update();
}

void AdvancedSettings::BufferSizes::Load()
{
    GSH::instance().set_file_buffer_size(this->file);
    GSH::instance().set_input_buffer_size(this->input);
    GSH::instance().set_output_buffer_size(this->output);
    GSH::instance().set_record_buffer_size(this->record);
    GSH::instance().set_time_transformation_cuts_output_buffer_size(this->time_transformation_cuts);
}

void AdvancedSettings::Filter2DSmooth::Load()
{
    GSH::instance().set_filter2d_smooth_low(this->low);
    GSH::instance().set_filter2d_smooth_high(this->high);
}

void AdvancedSettings::ContrastThreshold::Load()
{
    GSH::instance().set_contrast_lower_threshold(this->lower);
    GSH::instance().set_contrast_upper_threshold(this->upper);
    GSH::instance().set_cuts_contrast_p_offset(this->frame_index_offset);
}

void AdvancedSettings::Load()
{
    this->buffer_size.Load();
    this->filter2d_smooth.Load();
    this->contrast.Load();
    GSH::instance().set_renorm_constant(this->renorm_constant);
    GSH::instance().set_raw_bitshift(this->raw_bitshift);
}

void Composite::Load()
{
    GSH::instance().set_composite_kind(this->mode);
    GSH::instance().set_composite_auto_weights(this->auto_weight);
    GSH::instance().set_rgb(this->rgb);
    GSH::instance().set_hsv(this->hsv);
}

void ComputeSettings::Load()
{
    this->image_rendering.Load();
    this->color_composite_image.Load();
    this->advanced.Load();
    this->view.Load();
}

void Windows::Load()
{
    GSH::instance().set_xy(this->xy);
    GSH::instance().set_yz(this->yz);
    GSH::instance().set_xz(this->xz);
    GSH::instance().set_filter2d(this->filter2d);
}

void Reticle::Load()
{
    GSH::instance().set_reticle_display_enabled(this->display_enabled);
    GSH::instance().set_reticle_scale(this->scale);
}

void Views::Load()
{
    GSH::instance().set_img_type(this->image_type);
    GSH::instance().set_fft_shift_enabled(this->fft_shift);
    GSH::instance().set_x(this->x);
    GSH::instance().set_y(this->y);
    GSH::instance().set_p(this->z);
    GSH::instance().set_q(this->z2);
    this->window.Load();
    GSH::instance().set_renorm_enabled(this->renorm);
    this->reticle.Load();
}

void Rendering::Convolution::Load()
{
    GSH::instance().set_convolution_enabled(this->enabled);
    UserInterfaceDescriptor::instance().convo_name = this->type;
    GSH::instance().set_divide_convolution_enabled(this->divide);
}

void Rendering::Filter2D::Load()
{
    GSH::instance().set_filter2d_enabled(this->enabled);
    GSH::instance().set_filter2d_n1(this->inner_radius);
    GSH::instance().set_filter2d_n2(this->outer_radius);
}

void Rendering::Load()
{
    GSH::instance().set_time_stride(this->time_transformation_stride);
    GSH::instance().set_compute_mode(this->image_mode);
    GSH::instance().set_batch_size(this->batch_size);
    this->filter2d.Load();
    GSH::instance().set_space_transformation(this->space_transformation);
    GSH::instance().set_time_transformation(this->time_transformation);
    GSH::instance().set_time_transformation_size(this->time_transformation_size);
    GSH::instance().set_lambda(this->lambda);
    GSH::instance().set_z_distance(this->propagation_distance);
    this->convolution.Load();
}

void ComputeSettings::Dump(const std::string& filename)
{
    json compute_json;
    this->Update();
    to_json(compute_json, *this);

    auto path_path = std::filesystem::path(holovibes::settings::dump_filepath) / (filename + ".json");
    auto file_content = std::ofstream(path_path, std::ifstream::out);
    file_content << std::setw(1) << compute_json;
}

} // namespace holovibes
