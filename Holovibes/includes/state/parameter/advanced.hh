#pragma once

#include "custom_parameter.hh"

namespace holovibes
{

using DisplayRate = FloatParameter<30, "display_rate">;

using InputBufferSize = UIntParameter<512, "input_buffer_size">;

} // namespace holovibes
