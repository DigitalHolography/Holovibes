/*! \file
 *
 * \brief #TODO Add a description for this file
 */

#pragma once

#include "parameter.hh"
#include "micro_cache.hh"
#include "advanced_struct.hh"

namespace holovibes
{
// clang-format off

class DisplayRate : public FloatParameter<30, "display_rate">{};
//! \brief Max file buffer size
class FileBufferSize : public UIntParameter<512, "file_buffer_size">{};
//! \brief Max size of input queue in number of images.
class InputBufferSize : public UIntParameter<512, "input_buffer_size">{};
//! \brief Max size of output queue in number of images.
class OutputBufferSize : public UIntParameter<256, "output_buffer_size">{};
//! \brief Max size of frame record queue in number of images.
class RecordBufferSize : public UIntParameter<1024, "record_buffer_size">{};
class TimeTransformationCutsBufferSize : public UIntParameter<512, "time_transformation_cuts_buffer_size">{};

class Filter2DSmooth : public Parameter<Filter2DSmoothStruct, DefaultLiteral<Filter2DSmoothStruct>{}, "filter2d_smooth">{};
class ContrastThreshold : public Parameter<ContrastThresholdStruct, DefaultLiteral<ContrastThresholdStruct>{}, "contrast_threshold">{};
class RenormConstant : public UIntParameter<5, "renorm_constant">{};
class RawBitshift : public IntParameter<0, "raw_bitshift">{};

// clang-format on

using BasicAdvancedCache = MicroCache<DisplayRate,
                                      FileBufferSize,
                                      InputBufferSize,
                                      OutputBufferSize,
                                      RecordBufferSize,
                                      TimeTransformationCutsBufferSize,
                                      Filter2DSmooth,
                                      ContrastThreshold,
                                      RenormConstant,
                                      RawBitshift>;

// clang-format off

class AdvancedCache : public BasicAdvancedCache
{
  public:
    using Base = BasicAdvancedCache;
    class Cache : public Base::Cache{};
    class Ref : public Base::Ref{};
};

// clang-format on

} // namespace holovibes
