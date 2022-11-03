/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

#include "compute_struct.hh"
#include "enum_computation.hh"
#include "enum_time_transformation.hh"
#include "enum_space_transformation.hh"

namespace holovibes
{
// clang-format off

class BatchSize : public IntParameter<1, "batch_size">{};
class TimeStride : public IntParameter<1, "time_stride">{};
class Lambda : public FloatParameter<852e-9f, "lambda">{};
class TimeTransformationSize : public UIntParameter<1, "time_transformation_size">{};
class SpaceTransformation : public CustomParameter<SpaceTransformationEnum, SpaceTransformationEnum::NONE, "space_transformation", SpaceTransformationEnum>{};
class TimeTransformation : public CustomParameter<TimeTransformationEnum, TimeTransformationEnum::NONE, "time_transformation", TimeTransformationEnum>{};
class ZDistance : public FloatParameter<1.50f, "z_distance">{};
class Convolution_PARAM : public CustomParameter<ConvolutionStruct, DefaultLiteral<ConvolutionStruct>{}, "convolution">{};
class InputFps : public UIntParameter<60, "input_fps">{};
class ComputeMode_PARAM : public CustomParameter<Computation, Computation::Raw, "compute_mode", Computation>{};
class PixelSize : public FloatParameter<12.0f, "pixel_size">{};
class UnwrapHistorySize : public UIntParameter<1, "unwrap_history_stopped">{};
class Unwrap2DRequested : public BoolParameter<false, "unwrap_2d_requested">{};
class IsComputationStopped : public BoolParameter<true, "is_computation_stopped">{};
class TimeTransformationCutsOutputBufferSize : public UIntParameter<512, "time_transformation_cuts_output_buffer_size">{};
class TimeTransformationCutsEnable : public BoolParameter<false, "time_transformation_cuts_enable">{};

// clang-format on

class ComputeCache : public MicroCache<BatchSize,
                                       TimeStride,
                                       Lambda,
                                       TimeTransformationSize,
                                       SpaceTransformation,
                                       TimeTransformation,
                                       ZDistance,
                                       Convolution_PARAM,
                                       InputFps,
                                       ComputeMode_PARAM,
                                       PixelSize,
                                       UnwrapHistorySize,
                                       Unwrap2DRequested,
                                       IsComputationStopped,
                                       TimeTransformationCutsOutputBufferSize,
                                       TimeTransformationCutsEnable>
{
};

} // namespace holovibes
