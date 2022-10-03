#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class TimeTransformationStride : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 1;

  public:
    TimeTransformationStride()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    TimeTransformationStride(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "time_transformation_stride"; }
    const char* get_key() const override { return TimeTransformationStride::static_key(); }
};

} // namespace holovibes
