#pragma once

#include "custom_parameter.hh"

namespace holovibes
{

using BatchSize = IntParameter<1, "batch_size">;
using DivideConvolutionEnable = BoolParameter<false, "divide_convolution_enabled">;

} // namespace holovibes
