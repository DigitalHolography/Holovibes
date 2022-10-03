#pragma once

#include "custom_parameter.hh"
#include "enum_window_kind.hh"

namespace holovibes
{

class WindowKindParam : public ICustomParameter<WindowKind>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = WindowKind::XYview;

  public:
    WindowKindParam()
        : ICustomParameter<WindowKind>(DEFAULT_VALUE)
    {
    }

    WindowKindParam(TransfertType value)
        : ICustomParameter<WindowKind>(value)
    {
    }

  public:
    static const char* static_key() { return "window_kind"; }
    const char* get_key() const override { return WindowKindParam::static_key(); }
};

} // namespace holovibes
