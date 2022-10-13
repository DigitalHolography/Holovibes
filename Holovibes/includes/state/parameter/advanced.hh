#pragma once

#include "custom_parameter.hh"
#include "micro_cache_tmp.hh"

namespace holovibes
{

using DisplayRate = FloatParameter<30, "display_rate">;
using InputBufferSize = UIntParameter<512, "input_buffer_size">;
using OutputBufferSize = UIntParameter<256, "output_buffer_size">;
using RecordBufferSize = UIntParameter<1024, "record_buffer_size">;
using ContrastLowerThreshold = FloatParameter<0.5f, "contrast_lower_threshold">;
using ContrastUpperThreshold = FloatParameter<99.5f, "contrast_upper_threshold">;
using RawBitshift = IntParameter<0, "raw_bitshift">;
using RenormConstant = UIntParameter<5, "renorm_constant">;
using CutsContrastPOffset = UIntParameter<2, "cuts_contrast_p_offset">;

using AdvancedCache = MicroCacheTmp<DisplayRate,
                                    InputBufferSize,
                                    OutputBufferSize,
                                    RecordBufferSize,
                                    ContrastLowerThreshold,
                                    ContrastUpperThreshold,
                                    RawBitshift,
                                    RenormConstant,
                                    CutsContrastPOffset>;

} // namespace holovibes
