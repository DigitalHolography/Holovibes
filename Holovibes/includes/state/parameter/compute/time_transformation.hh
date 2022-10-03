#pragma once

#include "custom_parameter.hh"
#include "enum_time_transformation.hh"

namespace holovibes
{

class TimeTransformationParam : public ICustomParameter<TimeTransformation>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = TimeTransformation::NONE;

  public:
    TimeTransformationParam()
        : ICustomParameter<TimeTransformation>(DEFAULT_VALUE)
    {
    }

    TimeTransformationParam(TransfertType value)
        : ICustomParameter<TimeTransformation>(value)
    {
    }

  public:
    static const char* static_key() { return "time_transformation"; }
    const char* get_key() const override { return TimeTransformationParam::static_key(); }
};

} // namespace holovibes
