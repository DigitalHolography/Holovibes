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
    this->xy = GSH::instance().get_value<ViewXY>();
    this->yz = GSH::instance().get_value<ViewYZ>();
    this->xz = GSH::instance().get_value<ViewXZ>();
    this->filter2d = GSH::instance().get_value<ViewFilter2D>();
}

void Views::Update()
{
    this->img_type = GSH::instance().get_value<ImageType>();
    this->fft_shift = GSH::instance().get_value<FftShiftEnabled>();
    this->x = GSH::instance().get_value<ViewAccuX>();
    this->y = GSH::instance().get_value<ViewAccuY>();
    this->p = GSH::instance().get_value<ViewAccuP>();
    this->q = GSH::instance().get_value<ViewAccuQ>();
    this->window.Update();
    this->renorm = GSH::instance().get_value<RenormEnabled>();
    this->reticle = GSH::instance().get_value<Reticle>();
}

void Rendering::Update()
{
    this->image_mode = GSH::instance().get_value<ComputeMode>();
    this->batch_size = GSH::instance().get_value<BatchSize>();
    this->time_transformation_stride = GSH::instance().get_value<TimeStride>();
    this->filter2d = GSH::instance().get_value<Filter2D>();
    this->space_transformation = GSH::instance().get_value<SpaceTransformation>();
    this->time_transformation = GSH::instance().get_value<TimeTransformation>();
    this->time_transformation_size = GSH::instance().get_value<TimeTransformationSize>();
    this->lambda = GSH::instance().get_value<Lambda>();
    this->z_distance = GSH::instance().get_value<ZDistance>();
    this->convolution = GSH::instance().get_value<Convolution>();
}

void BufferSize::Update()
{
    this->file = GSH::instance().get_value<FileBufferSize>();
    this->input = GSH::instance().get_value<InputBufferSize>();
    this->output = GSH::instance().get_value<OutputBufferSize>();
    this->record = GSH::instance().get_value<RecordBufferSize>();
    this->time_transformation_cuts = GSH::instance().get_value<TimeTransformationCutsBufferSize>();
}

void AdvancedSettings::Update()
{
    this->buffer_size.Update();
    this->filter2d_smooth = GSH::instance().get_value<Filter2DSmooth>();
    this->contrast = GSH::instance().get_value<ContrastThreshold>();
    this->renorm_constant = GSH::instance().get_value<RenormConstant>();
    this->raw_bitshift = GSH::instance().get_value<RawBitshift>();
}

void Composite::Update()
{
    this->mode = GSH::instance().get_value<CompositeKind>();
    this->composite_auto_weights = GSH::instance().get_value<CompositeAutoWeights>();
    this->rgb = GSH::instance().get_value<CompositeRGB>();
    this->hsv = GSH::instance().get_value<CompositeHSV>();
}

void ComputeSettings::Update()
{
    this->image_rendering.Update();
    this->view.Update();
    this->composite.Update();
    this->advanced.Update();
}

void BufferSize::Load()
{
    GSH::instance().set_value<FileBufferSize>(this->file);
    GSH::instance().set_value<InputBufferSize>(this->input);
    GSH::instance().set_value<OutputBufferSize>(this->output);
    GSH::instance().set_value<RecordBufferSize>(this->record);
    GSH::instance().set_value<TimeTransformationCutsBufferSize>(this->time_transformation_cuts);
}

void AdvancedSettings::Load()
{
    this->buffer_size.Load();
    GSH::instance().set_value<Filter2DSmooth>(this->filter2d_smooth);
    GSH::instance().set_value<ContrastThreshold>(this->contrast);
    GSH::instance().set_value<RenormConstant>(this->renorm_constant);
    GSH::instance().set_value<RawBitshift>(this->raw_bitshift);
}

void Composite::Load()
{
    GSH::instance().set_value<CompositeKind>(this->mode);
    GSH::instance().set_value<CompositeAutoWeights>(this->composite_auto_weights);
    GSH::instance().set_value<CompositeRGB>(this->rgb);
    GSH::instance().set_value<CompositeHSV>(this->hsv);
}

void ComputeSettings::Load()
{
    this->image_rendering.Load();
    this->composite.Load();
    this->advanced.Load();
    this->view.Load();
}

void Windows::Load()
{
    GSH::instance().set_value<ViewXY>(this->xy);
    GSH::instance().set_value<ViewYZ>(this->yz);
    GSH::instance().set_value<ViewXZ>(this->xz);
    GSH::instance().set_value<ViewFilter2D>(this->filter2d);
}

void Views::Load()
{
    GSH::instance().set_value<ImageType>(this->img_type);
    GSH::instance().set_value<FftShiftEnabled>(this->fft_shift);
    GSH::instance().set_value<ViewAccuX>(this->x);
    GSH::instance().set_value<ViewAccuY>(this->y);
    GSH::instance().set_value<ViewAccuP>(this->p);
    GSH::instance().set_value<ViewAccuQ>(this->q);
    this->window.Load();
    GSH::instance().set_value<RenormEnabled>(this->renorm);
    GSH::instance().set_value<Reticle>(this->reticle);
}

/*
void Rendering::ConvolutionStruct::Load()
{
    GSH::instance().set_convolution_enabled(this->enabled);
    // FIXME : Check
    UserInterfaceDescriptor::instance().convo_name = this->type;
    GSH::instance().set_divide_convolution_enabled(this->divide);
}*/

void Rendering::Load()
{
    GSH::instance().set_value<ComputeMode>(this->image_mode);
    GSH::instance().set_value<BatchSize>(this->batch_size);
    GSH::instance().set_value<TimeStride>(this->time_transformation_stride);
    GSH::instance().set_value<Filter2D>(this->filter2d);
    GSH::instance().set_value<SpaceTransformation>(this->space_transformation);
    GSH::instance().set_value<TimeTransformation>(this->time_transformation);
    GSH::instance().set_value<TimeTransformationSize>(this->time_transformation_size);
    GSH::instance().set_value<Lambda>(this->lambda);
    GSH::instance().set_value<ZDistance>(this->z_distance);
    GSH::instance().set_value<Convolution>(this->convolution);
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