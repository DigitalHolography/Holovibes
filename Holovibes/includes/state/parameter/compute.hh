#pragma once

#include "custom_parameter.hh"

namespace holovibes
{

using BatchSize = IntParameter<1, "batch_size">;
using TimeStride = IntParameter<1, "time_stride">;

using DivideConvolutionEnable = BoolParameter<false, "divide_convolution_enabled">;

using Lambda = FloatParameter<852e-9f, "lambda">;

} // namespace holovibes
