#pragma once

#include "custom_parameter.hh"
#include "view_struct.hh"

namespace holovibes
{

class View_XZParam : public ICustomParameter<View_XZ>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = View_XYZ{};

  public:
    View_XZParam()
        : ICustomParameter<View_XZ>(DEFAULT_VALUE)
    {
    }

    View_XZParam(TransfertType value)
        : ICustomParameter<View_XZ>(value)
    {
    }

  public:
    static const char* static_key() { return "View_XZ"; }
    const char* get_key() const override { return View_XZParam::static_key(); }
};

} // namespace holovibes
