#pragma once

#include "custom_parameter.hh"
#include "view_struct.hh"

namespace holovibes
{

class View_XYParam : public ICustomParameter<View_XY>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = View_XYZ{};

  public:
    View_XYParam()
        : ICustomParameter<View_XY>(DEFAULT_VALUE)
    {
    }

    View_XYParam(TransfertType value)
        : ICustomParameter<View_XY>(value)
    {
    }

  public:
    static const char* static_key() { return "View_XY"; }
    const char* get_key() const override { return View_XYParam::static_key(); }
};

} // namespace holovibes
