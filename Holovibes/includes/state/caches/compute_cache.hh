/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

#include "enum_computation.hh"
#include "enum_time_transformation.hh"
#include "enum_space_transformation.hh"
#include "rendering_struct.hh"

namespace holovibes
{
// clang-format off

class ComputeMode : public CustomParameter<Computation, Computation::Raw, "compute_mode", Computation>{};
class BatchSize : public IntParameter<1, "batch_size">{};
class TimeStride : public IntParameter<1, "time_stride">{};
class Filter2D : public CustomParameter<Filter2DStruct, DefaultLiteral<Filter2DStruct>{}, "filter2d">{};
class SpaceTransformation : public CustomParameter<SpaceTransformationEnum, SpaceTransformationEnum::NONE, "space_transformation", SpaceTransformationEnum>{};
class TimeTransformation : public CustomParameter<TimeTransformationEnum, TimeTransformationEnum::NONE, "time_transformation", TimeTransformationEnum>{};
class TimeTransformationSize : public UIntParameter<1, "time_transformation_size">{};
class Lambda : public FloatParameter<852e-9f, "lambda">{};
class ZDistance : public FloatParameter<1.50f, "z_distance">{};
class Convolution : public CustomParameter<ConvolutionStruct, DefaultLiteral<ConvolutionStruct>{}, "convolution">{};

class InputFps : public UIntParameter<60, "input_fps">{};
class PixelSize : public FloatParameter<12.0f, "pixel_size">{};
class UnwrapHistorySize : public UIntParameter<1, "unwrap_history_stopped">{};
class Unwrap2DRequested : public BoolParameter<false, "unwrap_2d_requested">{};
class IsComputationStopped : public BoolParameter<true, "is_computation_stopped">{};
class TimeTransformationCutsEnable : public BoolParameter<false, "time_transformation_cuts_enable">{};

// clang-format on

class ComputeCache : public MicroCache<ComputeMode,
                                       BatchSize,
                                       TimeStride,
                                       Filter2D,
                                       SpaceTransformation,
                                       TimeTransformation,
                                       TimeTransformationSize,
                                       Lambda,
                                       ZDistance,
                                       Convolution,
                                       InputFps,
                                       PixelSize,
                                       UnwrapHistorySize,
                                       Unwrap2DRequested,
                                       IsComputationStopped,
                                       TimeTransformationCutsEnable>
{
};

} // namespace holovibes
