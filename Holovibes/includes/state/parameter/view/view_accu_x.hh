#pragma once

#include "custom_parameter.hh"
#include "view_struct.hh"

namespace holovibes
{

class View_Accu_XParam : public ICustomParameter<View_Accu_X>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = View_Accu_XY{};

  public:
    View_Accu_XParam()
        : ICustomParameter<View_Accu_X>(DEFAULT_VALUE)
    {
    }

    View_Accu_XParam(TransfertType value)
        : ICustomParameter<View_Accu_X>(value)
    {
    }

  public:
    static const char* static_key() { return "View_Accu_X"; }
    const char* get_key() const override { return View_Accu_XParam::static_key(); }
};

} // namespace holovibes
