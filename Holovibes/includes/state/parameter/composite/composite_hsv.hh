#pragma once

#include "composite_struct.hh"
#include "custom_parameter.hh"

namespace holovibes
{
class CompositeHSVParam : public ICustomParameter<Composite_HSV>
{
  public:
    static constexpr ValueType DEFAULT_VALUE;

  public:
    CompositeHSVParam()
        : ICustomParameter<Composite_HSV>(DEFAULT_VALUE)
    {
    }

    CompositeHSVParam(TransfertType value)
        : ICustomParameter<Composite_HSV>(value)
    {
    }

  public:
    static const char* static_key() { return "composite_hsv"; }
    const char* get_key() const override { return CompositeHSVParam::static_key(); }
};

} // namespace holovibes
