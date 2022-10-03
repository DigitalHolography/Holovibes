#pragma once

#include "custom_parameter.hh"
#include "view_struct.hh"

namespace holovibes
{

class View_YZParam : public ICustomParameter<View_YZ>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = View_XYZ{};

  public:
    View_YZParam()
        : ICustomParameter<View_YZ>(DEFAULT_VALUE)
    {
    }

    View_YZParam(TransfertType value)
        : ICustomParameter<View_YZ>(value)
    {
    }

  public:
    static const char* static_key() { return "View_YZ"; }
    const char* get_key() const override { return View_YZParam::static_key(); }
};

} // namespace holovibes
