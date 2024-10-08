/*!
 * \file all_struct.cc
 * \brief Contains the definition of `Update`, `Load`, `Dump` and `Assert` functions for all the structs.
 * - `Update` function updates the struct with the values from the GSH.
 * - `Load` function updates the GSH with the values from the struct.
 * - `Dump` function writes the struct to a json file.
 * - `Assert` function checks if the values of the struct are valid.
 */

#include <climits> // for UINT_MAX
#include <iomanip>
#include <filesystem>
#include <fstream>

#include "compute_settings_struct.hh"
#include "API.hh"
#include "all_struct.hh"

#define UPPER_BOUND(percentage) (UINT_MAX * percentage / 100)

#define GET_SETTING(setting) holovibes::Holovibes::instance().get_setting<holovibes::settings::setting>().value
#define UPDATE_SETTING(setting, value)                                                                                 \
    holovibes::Holovibes::instance().update_setting(holovibes::settings::setting{value})

namespace holovibes
{

void Windows::Update()
{
    this->xy = GET_SETTING(XY);
    this->yz = GET_SETTING(YZ);
    this->xz = GET_SETTING(XZ);
    this->filter2d = GET_SETTING(Filter2d);
}

void Reticle::Update()
{
    this->display_enabled = GET_SETTING(ReticleDisplayEnabled);
    this->scale = GET_SETTING(ReticleScale);
}

void Views::Update()
{
    this->image_type = GET_SETTING(ImageType);
    this->fft_shift = GET_SETTING(FftShiftEnabled);
    this->x = GET_SETTING(X); // GSH::instance().get_x();
    this->y = GET_SETTING(Y);
    this->z = GET_SETTING(P);
    this->z2 = GET_SETTING(Q);
    this->window.Update();
    this->renorm = GET_SETTING(RenormEnabled);
    this->reticle.Update();
}

void Rendering::Convolution::Update()
{
    this->enabled = GET_SETTING(ConvolutionEnabled);
    this->type = UserInterfaceDescriptor::instance().convo_name;
    this->divide = GET_SETTING(DivideConvolutionEnabled);
}

void Rendering::Filter::Update()
{
    this->enabled = api::get_filter_enabled();
    this->type = UserInterfaceDescriptor::instance().filter_name;
}

void Rendering::Filter2D::Update()
{
    this->enabled = GET_SETTING(Filter2dEnabled);
    this->inner_radius = GET_SETTING(Filter2dN1);
    this->outer_radius = GET_SETTING(Filter2dN2);
}

void Rendering::Update()
{
    this->image_mode = GET_SETTING(ComputeMode);
    this->frame_packet = GET_SETTING(FramePacket);
    this->batch_size = GET_SETTING(BatchSize);
    this->time_transformation_stride = GET_SETTING(TimeStride);
    this->filter2d.Update();
    this->space_transformation = GET_SETTING(SpaceTransformation);
    this->time_transformation = GET_SETTING(TimeTransformation);
    this->time_transformation_size = GET_SETTING(TimeTransformationSize);
    this->lambda = GET_SETTING(Lambda);
    this->propagation_distance = GET_SETTING(ZDistance);
    this->convolution.Update();
    this->input_filter.Update();
}

void AdvancedSettings::BufferSizes::Update()
{
    this->file = static_cast<unsigned int>(GET_SETTING(FileBufferSize));
    this->input = static_cast<unsigned int>(GET_SETTING(InputBufferSize));
    this->output = static_cast<unsigned int>(GET_SETTING(OutputBufferSize));
    this->record = static_cast<unsigned int>(GET_SETTING(RecordBufferSize));
    this->time_transformation_cuts = GET_SETTING(TimeTransformationCutsOutputBufferSize);
}

void AdvancedSettings::Filter2DSmooth::Update()
{
    this->low = GET_SETTING(Filter2dSmoothLow);
    this->high = GET_SETTING(Filter2dSmoothHigh);
}

void AdvancedSettings::ContrastThreshold::Update()
{
    this->lower = GET_SETTING(ContrastLowerThreshold);
    this->upper = GET_SETTING(ContrastUpperThreshold);
    this->frame_index_offset = static_cast<unsigned int>(GET_SETTING(CutsContrastPOffset));
}

void AdvancedSettings::Update()
{
    this->buffer_size.Update();
    this->filter2d_smooth.Update();
    this->contrast.Update();
    this->renorm_constant = GET_SETTING(RenormConstant);
    this->raw_bitshift = static_cast<unsigned int>(GET_SETTING(RawBitshift));
    // FIXME: optional value might not be that greate of an idea
    if (GET_SETTING(RecordFrameCount).has_value())
        this->nb_frames_to_record = static_cast<unsigned int>(GET_SETTING(RecordFrameCount).value());
}

void Composite::Update()
{
    this->mode = GET_SETTING(CompositeKind);
    this->auto_weight = GET_SETTING(CompositeAutoWeights);
    this->rgb = GET_SETTING(RGB);
    this->hsv = GET_SETTING(HSV);
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
    UPDATE_SETTING(FileBufferSize, this->file);
    UPDATE_SETTING(InputBufferSize, this->input);
    UPDATE_SETTING(OutputBufferSize, this->output);
    UPDATE_SETTING(RecordBufferSize, this->record);
    UPDATE_SETTING(TimeTransformationCutsOutputBufferSize, this->time_transformation_cuts);
}

void AdvancedSettings::Filter2DSmooth::Load()
{
    UPDATE_SETTING(Filter2dSmoothLow, this->low);
    UPDATE_SETTING(Filter2dSmoothHigh, this->high);
}

void AdvancedSettings::ContrastThreshold::Load()
{
    UPDATE_SETTING(ContrastLowerThreshold, this->lower);
    UPDATE_SETTING(ContrastUpperThreshold, this->upper);
    UPDATE_SETTING(CutsContrastPOffset, this->frame_index_offset);
}

void AdvancedSettings::Load()
{
    this->buffer_size.Load();
    this->filter2d_smooth.Load();
    this->contrast.Load();
    UPDATE_SETTING(RenormConstant, this->renorm_constant);
    UPDATE_SETTING(RawBitshift, this->raw_bitshift);
    if (this->nb_frames_to_record != 0)
        UPDATE_SETTING(RecordFrameCount, this->nb_frames_to_record);
}

void Composite::Load()
{
    UPDATE_SETTING(CompositeKind, this->mode);
    UPDATE_SETTING(CompositeAutoWeights, this->auto_weight);
    UPDATE_SETTING(RGB, this->rgb);
    UPDATE_SETTING(HSV, this->hsv);
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
    UPDATE_SETTING(XY, this->xy);
    UPDATE_SETTING(XZ, this->xz);
    UPDATE_SETTING(YZ, this->yz);
    UPDATE_SETTING(Filter2d, this->filter2d);
}

void Reticle::Load()
{
    UPDATE_SETTING(ReticleDisplayEnabled, this->display_enabled);
    UPDATE_SETTING(ReticleScale, this->scale);
}

void Views::Load()
{
    UPDATE_SETTING(ImageType, this->image_type);
    UPDATE_SETTING(FftShiftEnabled, this->fft_shift);
    UPDATE_SETTING(X, this->x);
    UPDATE_SETTING(Y, this->y);
    UPDATE_SETTING(P, this->z);
    UPDATE_SETTING(Q, this->z2);
    this->window.Load();
    UPDATE_SETTING(RenormEnabled, this->renorm);
    this->reticle.Load();
}

void Rendering::Convolution::Load()
{
    UPDATE_SETTING(ConvolutionEnabled, this->enabled);
    UserInterfaceDescriptor::instance().convo_name = this->type;
    UPDATE_SETTING(DivideConvolutionEnabled, this->divide);
}

void Rendering::Filter::Load()
{
    UPDATE_SETTING(FilterEnabled, this->enabled && this->type != UID_FILTER_TYPE_DEFAULT);
    UserInterfaceDescriptor::instance().filter_name = this->type;
}

void Rendering::Filter2D::Load()
{
    UPDATE_SETTING(Filter2dEnabled, this->enabled);
    UPDATE_SETTING(Filter2dN1, this->inner_radius);
    UPDATE_SETTING(Filter2dN2, this->outer_radius);
}

void Rendering::Load()
{
    UPDATE_SETTING(TimeStride, this->time_transformation_stride);
    UPDATE_SETTING(ComputeMode, this->image_mode);
    api::set_batch_size(this->batch_size);
    this->filter2d.Load();
    UPDATE_SETTING(SpaceTransformation, this->space_transformation);
    UPDATE_SETTING(TimeTransformation, this->time_transformation);
    UPDATE_SETTING(TimeTransformationSize, this->time_transformation_size);
    UPDATE_SETTING(Lambda, this->lambda);
    UPDATE_SETTING(ZDistance, this->propagation_distance);
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

void Rendering::Convolution::Assert() const
{
    /* if (this->enabled && this->type.empty())
        throw std::exception("Convolution type is empty");
    if (this->divide && !this->enabled)
        throw std::exception("Divide convolution is enabled but convolution is not"); */  // TODO: check if divide convolution can be enabled when convolution is disabled
}

void Rendering::Filter::Assert() const
{
    /* if (this->enabled && this->type.empty())
        throw std::exception("Filter type is empty");
    if (!this->enabled && !this->type.empty())
        throw std::exception("Filter type is not empty but filter is disabled"); */  // TODO: check if filter type can be empty when filter is disabled
}

void Rendering::Filter2D::Assert() const
{
    if (this->inner_radius >= this->outer_radius)
        throw std::exception("Inner radius is greater than outer radius");
}

void Rendering::Assert() const // TODO: check for a more appropriate upper bound, abose is probably a negative value
                               // that was flipped to positive
{
    if (this->time_transformation_stride == 0 ||
        this->time_transformation_stride > UPPER_BOUND(0.01)) // UPPER_BOUND(0.01) = ~429 000+
        throw std::exception("Time transformation stride value is invalid");
    if (this->batch_size == 0 || this->batch_size > UPPER_BOUND(0.01))
        throw std::exception("Batch size is value is invalid");
    if (this->time_transformation_size == 0 || this->time_transformation_size > UPPER_BOUND(0.01))
        throw std::exception("Time transformation size is 0");
    // TODO: does lamba and propagation distance have to be positive/have borbidden values ?
    this->filter2d.Assert();
    this->convolution.Assert();
    this->input_filter.Assert();
}

void AdvancedSettings::BufferSizes::Assert() const
{
    if (this->file == 0 || this->file > UPPER_BOUND(0.01)) // TODO: check for a more appropriate upper bound
        throw std::exception("File buffer size is invalid");
    if (this->input == 0 || this->input > UPPER_BOUND(0.01))
        throw std::exception("Input buffer size is invalid");
    if (this->output == 0 || this->output > UPPER_BOUND(0.01))
        throw std::exception("Output buffer size is invalid");
    if (this->record == 0 || this->record > UPPER_BOUND(0.01))
        throw std::exception("Record buffer size is invalid");
    if (this->time_transformation_cuts == 0 || this->time_transformation_cuts > UPPER_BOUND(0.01))
        throw std::exception("Time transformation cuts buffer size is invalid");
}

void AdvancedSettings::Filter2DSmooth::Assert() const
{
    if (this->low < 0 || this->low > 100) // TODO: check for a more appropriate upper bound
        throw std::exception("Low filter 2d smooth value is invalid");
    if (this->high < 0 || this->high > 100)
        throw std::exception("High filter 2d smooth value is invalid");
}

void AdvancedSettings::ContrastThreshold::Assert() const
{
    if (this->lower < 0 || this->lower > 100)
        throw std::exception("Lower contrast threshold value is invalid");
    if (this->upper < 0 || this->upper > 100)
        throw std::exception("Upper contrast threshold value is invalid");
    /* if (this->frame_index_offset < 0)
        throw std::exception("Frame index offset negative"); */  // TODO: check if frame index offset can be negative
}

void AdvancedSettings::Assert() const
{
    if (this->renorm_constant == 0)
        throw std::exception("Renorm constant is 0");
    if (this->raw_bitshift < 0)
        throw std::exception("Raw bitshift is negative");

    this->buffer_size.Assert();
    this->filter2d_smooth.Assert();
    this->contrast.Assert();
}

void Composite::Assert() const
{
    /*
    if (this->mode == Computation::RGB && this->auto_weight)
        throw std::exception("Auto weight is enabled but composite mode is RGB");
    if (this->mode == Computation::HSV && this->auto_weight)
        throw std::exception("Auto weight is enabled but composite mode is HSV");
    */  // TODO: check if auto weight can be enabled when composite mode is RGB or HSV
}

void ComputeSettings::Assert() const
{
    this->image_rendering.Assert();
    this->view.Assert();
    this->color_composite_image.Assert();
    this->advanced.Assert();
}

void Windows::Assert() const
{
    // TODO: check if xy, yz, xz and filter2d have to be positive/have forbidden values
}

void Reticle::Assert() const
{
    if (this->scale <= 0)
        throw std::exception("Reticle scale is 0 or negative");
}

void Views::Assert() const
{
    this->window.Assert();
    this->reticle.Assert();
}

} // namespace holovibes
