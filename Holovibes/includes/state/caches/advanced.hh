/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{

using DisplayRate = FloatParameter<30, "display_rate">;
using InputBufferSize = UIntParameter<512, "input_buffer_size">;
//! \brief Max size of output queue in number of images.
using OutputBufferSize = UIntParameter<256, "output_buffer_size">;
//! \brief Max size of frame record queue in number of images.
using RecordBufferSize = UIntParameter<1024, "record_buffer_size">;
using ContrastLowerThreshold = FloatParameter<0.5f, "contrast_lower_threshold">;
using ContrastUpperThreshold = FloatParameter<99.5f, "contrast_upper_threshold">;
using RawBitshift = IntParameter<0, "raw_bitshift">;
using RenormConstant = UIntParameter<5, "renorm_constant">;
using CutsContrastPOffset = UIntParameter<2, "cuts_contrast_p_offset">;

using AdvancedCache = MicroCache<DisplayRate,
                                 InputBufferSize,
                                 OutputBufferSize,
                                 RecordBufferSize,
                                 ContrastLowerThreshold,
                                 ContrastUpperThreshold,
                                 RawBitshift,
                                 RenormConstant,
                                 CutsContrastPOffset>;

} // namespace holovibes
