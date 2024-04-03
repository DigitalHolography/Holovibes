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
    this->xy = holovibes::Holovibes::instance().get_setting<settings::XY>().value;
    this->yz = holovibes::Holovibes::instance().get_setting<settings::YZ>().value;
    this->xz = holovibes::Holovibes::instance().get_setting<settings::XZ>().value;
    this->filter2d = holovibes::Holovibes::instance().get_setting<settings::Filter2d>().value;
}

void Reticle::Update()
{
    this->display_enabled =
        holovibes::Holovibes::instance().get_setting<settings::ReticleDisplayEnabled>().value;
    this->scale = holovibes::Holovibes::instance().get_setting<settings::ReticleScale>().value;
}

void Views::Update()
{
    this->image_type = holovibes::Holovibes::instance().get_setting<settings::ImageType>().value;
    this->fft_shift = holovibes::Holovibes::instance().get_setting<settings::FftShiftEnabled>().value;
    this->x = holovibes::Holovibes::instance().get_setting<settings::X>().value; // GSH::instance().get_x();
    this->y = holovibes::Holovibes::instance().get_setting<settings::Y>().value;
    this->z = holovibes::Holovibes::instance().get_setting<settings::P>().value;
    this->z2 = holovibes::Holovibes::instance().get_setting<settings::Q>().value;
    this->window.Update();
    this->renorm = holovibes::Holovibes::instance().get_setting<settings::RenormEnabled>().value;
    this->reticle.Update();
}

void Rendering::Convolution::Update()
{
    this->enabled = holovibes::Holovibes::instance().get_setting<settings::ConvolutionEnabled>().value;
    this->type = UserInterfaceDescriptor::instance().convo_name;
    this->divide = holovibes::Holovibes::instance().get_setting<settings::DivideConvolutionEnabled>().value;
}

void Rendering::Filter::Update()
{
    this->enabled = api::get_filter_enabled();
    this->type = UserInterfaceDescriptor::instance().filter_name;
}

void Rendering::Filter2D::Update()
{
    this->enabled = holovibes::Holovibes::instance().get_setting<settings::Filter2dEnabled>().value;
    this->inner_radius = holovibes::Holovibes::instance().get_setting<settings::Filter2dN1>().value;
    this->outer_radius = holovibes::Holovibes::instance().get_setting<settings::Filter2dN2>().value;
}

void Rendering::Update()
{
    this->image_mode = holovibes::Holovibes::instance().get_setting<settings::ComputeMode>().value;
    this->batch_size = holovibes::Holovibes::instance().get_setting<settings::BatchSize>().value;
    this->time_transformation_stride = holovibes::Holovibes::instance().get_setting<settings::TimeStride>().value;
    this->filter2d.Update();
    this->space_transformation = holovibes::Holovibes::instance().get_setting<settings::SpaceTransformation>().value;
    this->time_transformation = holovibes::Holovibes::instance().get_setting<settings::TimeTransformation>().value;
    this->time_transformation_size = holovibes::Holovibes::instance().get_setting<settings::TimeTransformationSize>().value;
    this->lambda = holovibes::Holovibes::instance().get_setting<settings::Lambda>().value;
    this->propagation_distance = holovibes::Holovibes::instance().get_setting<settings::ZDistance>().value;
    this->convolution.Update();
    this->input_filter.Update();
}

void AdvancedSettings::BufferSizes::Update()
{
    this->file = holovibes::Holovibes::instance().get_setting<settings::FileBufferSize>().value;
    this->input = holovibes::Holovibes::instance().get_setting<settings::InputBufferSize>().value;
    this->output = holovibes::Holovibes::instance().get_setting<settings::OutputBufferSize>().value;
    this->record = holovibes::Holovibes::instance().get_setting<settings::RecordBufferSize>().value;
    this->time_transformation_cuts = holovibes::Holovibes::instance().get_setting<settings::TimeTransformationCutsOutputBufferSize>().value;
}

void AdvancedSettings::Filter2DSmooth::Update()
{
    this->low = holovibes::Holovibes::instance().get_setting<settings::Filter2dSmoothLow>().value;
    this->high = holovibes::Holovibes::instance().get_setting<settings::Filter2dSmoothHigh>().value;
}

void AdvancedSettings::ContrastThreshold::Update()
{
    this->lower = holovibes::Holovibes::instance().get_setting<settings::ContrastLowerThreshold>().value;
    this->upper = holovibes::Holovibes::instance().get_setting<settings::ContrastUpperThreshold>().value;
    this->frame_index_offset = holovibes::Holovibes::instance().get_setting<settings::CutsContrastPOffset>().value;
}

void AdvancedSettings::Update()
{
    this->buffer_size.Update();
    this->filter2d_smooth.Update();
    this->contrast.Update();
    this->renorm_constant = holovibes::Holovibes::instance().get_setting<settings::RenormConstant>().value;
    this->raw_bitshift = holovibes::Holovibes::instance().get_setting<settings::RawBitshift>().value;
}

void Composite::Update()
{
    this->mode = holovibes::Holovibes::instance().get_setting<settings::CompositeKind>().value;
    this->auto_weight = holovibes::Holovibes::instance().get_setting<settings::CompositeAutoWeights>().value;
    this->rgb = holovibes::Holovibes::instance().get_setting<settings::RGB>().value;
    this->hsv = holovibes::Holovibes::instance().get_setting<settings::HSV>().value;
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
    Holovibes::instance().update_setting(settings::FileBufferSize{this->file});
    Holovibes::instance().update_setting(settings::InputBufferSize{this->input});
    Holovibes::instance().update_setting(settings::OutputBufferSize{this->output});
    Holovibes::instance().update_setting(settings::RecordBufferSize{this->record});
    Holovibes::instance().update_setting(settings::TimeTransformationCutsOutputBufferSize{this->time_transformation_cuts});
}

void AdvancedSettings::Filter2DSmooth::Load()
{
    Holovibes::instance().update_setting(settings::Filter2dSmoothLow{this->low});
    Holovibes::instance().update_setting(settings::Filter2dSmoothHigh{this->high});
}

void AdvancedSettings::ContrastThreshold::Load()
{
    Holovibes::instance().update_setting(settings::ContrastLowerThreshold{this->lower});
    Holovibes::instance().update_setting(settings::ContrastUpperThreshold{this->upper});
    Holovibes::instance().update_setting(settings::CutsContrastPOffset{this->frame_index_offset});
}

void AdvancedSettings::Load()
{
    this->buffer_size.Load();
    this->filter2d_smooth.Load();
    this->contrast.Load();
    Holovibes::instance().update_setting(settings::RenormConstant{this->renorm_constant});
    Holovibes::instance().update_setting(settings::RawBitshift{this->raw_bitshift});
}

void Composite::Load()
{
    Holovibes::instance().update_setting(settings::CompositeKind{this->mode});
    Holovibes::instance().update_setting(settings::CompositeAutoWeights{this->auto_weight});
    Holovibes::instance().update_setting(settings::RGB{this->rgb});
    Holovibes::instance().update_setting(settings::HSV{this->hsv});
}

void ComputeSettings::Load()
{
    this->advanced.Load();
    this->image_rendering.Load();
    this->color_composite_image.Load();
    this->view.Load();
}

void Windows::Load()
{
    holovibes::Holovibes::instance().update_setting(settings::XY{this->xy});
    holovibes::Holovibes::instance().update_setting(settings::XZ{this->xz});
    holovibes::Holovibes::instance().update_setting(settings::YZ{this->yz});
    holovibes::Holovibes::instance().update_setting(settings::Filter2d{this->filter2d});
}

void Reticle::Load()
{
    holovibes::Holovibes::instance().update_setting(settings::ReticleDisplayEnabled{this->display_enabled});
    holovibes::Holovibes::instance().update_setting(settings::ReticleScale{this->scale});
}

void Views::Load()
{
    holovibes::Holovibes::instance().update_setting(settings::ImageType{this->image_type});
    holovibes::Holovibes::instance().update_setting(settings::FftShiftEnabled{this->fft_shift});
    holovibes::Holovibes::instance().update_setting(settings::X{this->x});
    holovibes::Holovibes::instance().update_setting(settings::Y{this->y});
    holovibes::Holovibes::instance().update_setting(settings::P{this->z});
    holovibes::Holovibes::instance().update_setting(settings::Q{this->z2});
    this->window.Load();
    holovibes::Holovibes::instance().update_setting(settings::RenormEnabled{this->renorm});
    this->reticle.Load();
}

void Rendering::Convolution::Load()
{
    holovibes::Holovibes::instance().update_setting(settings::ConvolutionEnabled{this->enabled});
    UserInterfaceDescriptor::instance().convo_name = this->type;
    holovibes::Holovibes::instance().update_setting(settings::DivideConvolutionEnabled{this->divide});
}

void Rendering::Filter::Load()
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::FilterEnabled{this->enabled && this->type != UID_FILTER_TYPE_DEFAULT});
    UserInterfaceDescriptor::instance().filter_name = this->type;
}

void Rendering::Filter2D::Load()
{
    holovibes::Holovibes::instance().update_setting(settings::Filter2dEnabled{this->enabled});
    holovibes::Holovibes::instance().update_setting(settings::Filter2dN1{this->inner_radius});
    holovibes::Holovibes::instance().update_setting(settings::Filter2dN2{this->outer_radius});
}

void Rendering::Load()
{
    holovibes::Holovibes::instance().update_setting(settings::TimeStride{this->time_transformation_stride});
    holovibes::Holovibes::instance().update_setting(settings::ComputeMode{this->image_mode});
    api::set_batch_size(this->batch_size);
    this->filter2d.Load();
    holovibes::Holovibes::instance().update_setting(settings::SpaceTransformation{this->space_transformation});
    holovibes::Holovibes::instance().update_setting(settings::TimeTransformation{this->time_transformation});
    holovibes::Holovibes::instance().update_setting(settings::TimeTransformationSize{this->time_transformation_size});
    holovibes::Holovibes::instance().update_setting(settings::Lambda{this->lambda});
    holovibes::Holovibes::instance().update_setting(settings::ZDistance{this->propagation_distance});
    this->convolution.Load();
    this->input_filter.Load();
}

void ComputeSettings::Dump(const std::string& filename)
{
    json compute_json;
    this->Update();
    to_json(compute_json, *this);

    auto path_path = std::filesystem::path(settings::patch_dirpath) / (filename + ".json");
    auto file_content = std::ofstream(path_path, std::ifstream::out);
    file_content << std::setw(1) << compute_json;
}

} // namespace holovibes
