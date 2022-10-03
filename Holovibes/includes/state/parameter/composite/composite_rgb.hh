#pragma once

#include "composite_struct.hh"
#include "custom_parameter.hh"

namespace holovibes
{
class CompositeRGBParam : public ICustomParameter<Composite_RGB>
{
  public:
    static constexpr ValueType DEFAULT_VALUE;

  public:
    CompositeRGBParam()
        : ICustomParameter<Composite_RGB>(DEFAULT_VALUE)
    {
    }

    CompositeRGBParam(TransfertType value)
        : ICustomParameter<Composite_RGB>(value)
    {
    }

  public:
    static const char* static_key() { return "composite_rgb"; }
    const char* get_key() const override { return CompositeRGBParam::static_key(); }
};

} // namespace holovibes
