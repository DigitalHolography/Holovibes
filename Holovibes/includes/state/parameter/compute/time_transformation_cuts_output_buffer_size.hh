#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class TimeTransformationCutsOutputBufferSize : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 512;

  public:
    TimeTransformationCutsOutputBufferSize()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    TimeTransformationCutsOutputBufferSize(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "time_transformation_cuts_output_buffer_size"; }
    const char* get_key() const override { return TimeTransformationCutsOutputBufferSize::static_key(); }
};

} // namespace holovibes
