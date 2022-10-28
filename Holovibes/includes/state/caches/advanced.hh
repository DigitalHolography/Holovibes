/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "custom_parameter.hh"
#include "micro_cache.hh"

namespace holovibes
{
// clang-format off

class DisplayRate : public FloatParameter<30, "display_rate">{};
class InputBufferSize : public UIntParameter<512, "input_buffer_size">{};
//! \brief Max size of output queue in number of images.
class OutputBufferSize : public UIntParameter<256, "output_buffer_size">{};
//! \brief Max size of frame record queue in number of images.
class RecordBufferSize : public UIntParameter<1024, "record_buffer_size">{};
class ContrastLowerThreshold : public FloatParameter<0.5f, "contrast_lower_threshold">{};
class ContrastUpperThreshold : public FloatParameter<99.5f, "contrast_upper_threshold">{};
class RawBitshift : public IntParameter<0, "raw_bitshift">{};
class RenormConstant : public UIntParameter<5, "renorm_constant">{};
class CutsContrastPOffset : public UIntParameter<2, "cuts_contrast_p_offset">{};

// clang-format on

class AdvancedCache : public MicroCache<DisplayRate,
                                        InputBufferSize,
                                        OutputBufferSize,
                                        RecordBufferSize,
                                        ContrastLowerThreshold,
                                        ContrastUpperThreshold,
                                        RawBitshift,
                                        RenormConstant,
                                        CutsContrastPOffset>
{
};

} // namespace holovibes
