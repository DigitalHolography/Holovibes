#pragma once

#include "custom_parameter.hh"
#include "view_struct.hh"

namespace holovibes
{

class View_WindowParam : public ICustomParameter<View_Window>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = View_Window{};

  public:
    View_WindowParam()
        : ICustomParameter<View_Window>(DEFAULT_VALUE)
    {
    }

    View_WindowParam(TransfertType value)
        : ICustomParameter<View_Window>(value)
    {
    }

  public:
    static const char* static_key() { return "view_window"; }
    const char* get_key() const override { return View_WindowParam::static_key(); }
};

} // namespace holovibes
