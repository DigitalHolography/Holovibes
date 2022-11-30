/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "parameter.hh"
#include "micro_cache.hh"

#include "enum_compute_mode.hh"
#include "enum_time_transformation.hh"
#include "enum_space_transformation.hh"
#include "enum_img_type.hh"
#include "rendering_struct.hh"

namespace holovibes
{
// clang-format off

class ComputeMode : public Parameter<ComputeModeEnum, ComputeModeEnum::Raw, "compute_mode", ComputeModeEnum>{};
class ImageType : public Parameter<ImageTypeEnum, ImageTypeEnum::Modulus, "img_type", ImageTypeEnum>{};
class BatchSize : public IntParameter<1, "batch_size">{};
class TimeStride : public IntParameter<1, "time_stride">{};
class Filter2D : public Parameter<Filter2DStruct, DefaultLiteral<Filter2DStruct>{}, "filter2d">{};
class SpaceTransformation : public Parameter<SpaceTransformationEnum, SpaceTransformationEnum::NONE, "space_transformation", SpaceTransformationEnum>{};
class TimeTransformation : public Parameter<TimeTransformationEnum, TimeTransformationEnum::NONE, "time_transformation", TimeTransformationEnum>{};
class TimeTransformationSize : public UIntParameter<1, "time_transformation_size">{};
class Lambda : public FloatParameter<852e-9f, "lambda">{};
class ZDistance : public FloatParameter<1.50f, "z_distance">{};
class Convolution : public Parameter<ConvolutionStruct, DefaultLiteral<ConvolutionStruct>{}, "convolution">{};

class PixelSize : public FloatParameter<12.0f, "pixel_size">{};
class UnwrapHistorySize : public UIntParameter<1, "unwrap_history_stopped">{};
class Unwrap2DRequested : public BoolParameter<false, "unwrap_2d_requested">{};
class TimeTransformationCutsEnable : public BoolParameter<false, "time_transformation_cuts_enable">{};

// clang-format on

using ComputeCache = MicroCache<ComputeMode,
                                ImageType,
                                BatchSize,
                                TimeStride,
                                Filter2D,
                                SpaceTransformation,
                                TimeTransformation,
                                TimeTransformationSize,
                                Lambda,
                                ZDistance,
                                Convolution,
                                PixelSize,
                                UnwrapHistorySize,
                                Unwrap2DRequested,
                                TimeTransformationCutsEnable>;

} // namespace holovibes
