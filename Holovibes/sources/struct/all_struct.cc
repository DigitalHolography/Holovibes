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
    this->xy = api::detail::get_value<ViewXY>();
    this->yz = api::detail::get_value<ViewYZ>();
    this->xz = api::detail::get_value<ViewXZ>();
    this->filter2d = api::detail::get_value<ViewFilter2D>();
}

void Views::Update()
{
    this->image_type = api::detail::get_value<ImageType>();
    this->fft_shift = api::detail::get_value<FftShiftEnabled>();
    this->x = api::detail::get_value<ViewAccuX>();
    this->y = api::detail::get_value<ViewAccuY>();
    this->z = api::detail::get_value<ViewAccuP>();
    this->z2 = api::detail::get_value<ViewAccuQ>();
    this->window.Update();
    this->renorm = api::detail::get_value<RenormEnabled>();
    this->reticle = api::detail::get_value<Reticle>();
}

void Rendering::Update()
{
    this->image_mode = api::detail::get_value<ComputeMode>();
    this->batch_size = api::detail::get_value<BatchSize>();
    this->time_transformation_stride = api::detail::get_value<TimeStride>();
    this->filter2d = api::detail::get_value<Filter2D>();
    this->space_transformation = api::detail::get_value<SpaceTransformation>();
    this->time_transformation = api::detail::get_value<TimeTransformation>();
    this->time_transformation_size = api::detail::get_value<TimeTransformationSize>();
    this->lambda = api::detail::get_value<Lambda>();
    this->propagation_distance = api::detail::get_value<ZDistance>();
    this->convolution = api::detail::get_value<Convolution>();
}

void BufferSize::Update()
{
    this->file = api::detail::get_value<FileBufferSize>();
    this->input = api::detail::get_value<InputBufferSize>();
    this->output = api::detail::get_value<OutputBufferSize>();
    this->record = api::detail::get_value<RecordBufferSize>();
    this->time_transformation_cuts = api::detail::get_value<TimeTransformationCutsBufferSize>();
}

void AdvancedSettings::Update()
{
    this->buffer_size.Update();
    this->filter2d_smooth = api::detail::get_value<Filter2DSmooth>();
    this->contrast = api::detail::get_value<ContrastThreshold>();
    this->renorm_constant = api::detail::get_value<RenormConstant>();
    this->raw_bitshift = api::detail::get_value<RawBitshift>();
}

void Composite::Update()
{
    this->mode = api::detail::get_value<CompositeKind>();
    this->auto_weight = api::detail::get_value<CompositeAutoWeights>();
    this->rgb = api::detail::get_value<CompositeRGB>();
    this->hsv = api::detail::get_value<CompositeHSV>();
}

void ComputeSettings::Update()
{
    this->image_rendering.Update();
    this->view.Update();
    this->color_composite_image.Update();
    this->advanced.Update();
}

void BufferSize::Load()
{
    api::detail::set_value<FileBufferSize>(this->file);
    api::detail::set_value<InputBufferSize>(this->input);
    api::detail::set_value<OutputBufferSize>(this->output);
    api::detail::set_value<RecordBufferSize>(this->record);
    api::detail::set_value<TimeTransformationCutsBufferSize>(this->time_transformation_cuts);
}

void AdvancedSettings::Load()
{
    this->buffer_size.Load();
    api::detail::set_value<Filter2DSmooth>(this->filter2d_smooth);
    api::detail::set_value<ContrastThreshold>(this->contrast);
    api::detail::set_value<RenormConstant>(this->renorm_constant);
    api::detail::set_value<RawBitshift>(this->raw_bitshift);
}

void Composite::Load()
{
    api::detail::set_value<CompositeKind>(this->mode);
    api::detail::set_value<CompositeAutoWeights>(this->auto_weight);
    api::detail::set_value<CompositeRGB>(this->rgb);
    api::detail::set_value<CompositeHSV>(this->hsv);
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
    api::detail::set_value<ViewXY>(this->xy);
    api::detail::set_value<ViewYZ>(this->yz);
    api::detail::set_value<ViewXZ>(this->xz);
    api::detail::set_value<ViewFilter2D>(this->filter2d);
}

void Views::Load()
{
    api::detail::set_value<ImageType>(this->image_type);
    api::detail::set_value<FftShiftEnabled>(this->fft_shift);
    api::detail::set_value<ViewAccuX>(this->x);
    api::detail::set_value<ViewAccuY>(this->y);
    api::detail::set_value<ViewAccuP>(this->z);
    api::detail::set_value<ViewAccuQ>(this->z2);
    this->window.Load();
    api::detail::set_value<RenormEnabled>(this->renorm);
    api::detail::set_value<Reticle>(this->reticle);
}

void Rendering::Load()
{
    api::detail::set_value<ComputeMode>(this->image_mode);
    api::detail::set_value<BatchSize>(this->batch_size);
    api::detail::set_value<TimeStride>(this->time_transformation_stride);
    api::detail::set_value<Filter2D>(this->filter2d);
    api::detail::set_value<SpaceTransformation>(this->space_transformation);
    api::detail::set_value<TimeTransformation>(this->time_transformation);
    api::detail::set_value<TimeTransformationSize>(this->time_transformation_size);
    api::detail::set_value<Lambda>(this->lambda);
    api::detail::set_value<ZDistance>(this->propagation_distance);
    api::detail::set_value<Convolution>(this->convolution);
}

void ComputeSettings::Dump(const std::string& filename)
{
    json compute_json;
    this->Update();
    to_json(compute_json, *this);

    auto path_path = std::filesystem::path(holovibes::settings::patch_dirpath) / (filename + ".json");
    auto file_content = std::ofstream(path_path, std::ifstream::out);
    file_content << std::setw(1) << compute_json;
}

} // namespace holovibes
