#pragma once

#include "enum_composite_kind.hh"
#include "custom_parameter.hh"

namespace holovibes
{
class CompositeKind : public ICustomParameter<CompositeKind>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = CompositeKind::RGB;

  public:
    CompositeKind()
        : ICustomParameter<CompositeKind>(DEFAULT_VALUE)
    {
    }

    CompositeKind(TransfertType value)
        : ICustomParameter<CompositeKind>(value)
    {
    }

  public:
    static const char* static_key() { return "composite_kind"; }
    const char* get_key() const override { return CompositeKind::static_key(); }
};

} // namespace holovibes
