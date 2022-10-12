#pragma once

#include "custom_parameter.hh"
#include "micro_cache_tmp.hh"

namespace holovibes
{

//   (uint, output_buffer_size, 256),
//                            (uint, record_buffer_size, 1024),
//                            (float, contrast_lower_threshold, 0.5f),
//                            (int, raw_bitshift, 0),
//                            (float, contrast_upper_threshold, 99.5f),
//                            (unsigned, renorm_constant, 5),
//                            (uint, cuts_contrast_p_offset, 2));

using DisplayRate = FloatParameter<30, "display_rate">;
using InputBufferSize = UIntParameter<512, "input_buffer_size">;

using AdvancedCacheTmp = MicroCacheTmp<DisplayRate, InputBufferSize>;

} // namespace holovibes
