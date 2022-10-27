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

using BatchSize = IntParameter<1, "batch_size">;
using TimeStride = IntParameter<1, "time_stride">;
using Lambda = FloatParameter<852e-9f, "lambda">;
using TimeTransformationSize = UIntParameter<1, "time_transformation_size">;
using SpaceTransformationParam =
    CustomParameter<SpaceTransformation, SpaceTransformation::NONE, "space_transformation", SpaceTransformation>;
using TimeTransformationParam =
    CustomParameter<TimeTransformation, TimeTransformation::NONE, "time_transformation", TimeTransformation>;
using ZDistance = FloatParameter<1.50f, "z_distance">;
using Convolution = CustomParameter<ConvolutionStruct, DefaultLiteral<ConvolutionStruct>{}, "convolution">;
using InputFps = UIntParameter<60, "input_fps">;
using ComputeMode = CustomParameter<Computation, Computation::Raw, "compute_mode", Computation>;
using PixelSize = FloatParameter<12.0f, "pixel_size">;
using UnwrapHistorySize = UIntParameter<1, "unwrap_history_stopped">;
using Unwrap2DRequested = BoolParameter<false, "unwrap_2d_requested">;
using IsComputationStopped = BoolParameter<true, "is_computation_stopped">;
using TimeTransformationCutsOutputBufferSize = UIntParameter<512, "time_transformation_cuts_output_buffer_size">;
using TimeTransformationCutsEnable = BoolParameter<false, "time_transformation_cuts_enable">;

using ComputeCache = MicroCache<BatchSize,
                                TimeStride,
                                Lambda,
                                TimeTransformationSize,
                                SpaceTransformationParam,
                                TimeTransformationParam,
                                ZDistance,
                                Convolution,
                                InputFps,
                                ComputeMode,
                                PixelSize,
                                UnwrapHistorySize,
                                Unwrap2DRequested,
                                IsComputationStopped,
                                TimeTransformationCutsOutputBufferSize,
                                TimeTransformationCutsEnable>;

} // namespace holovibes
