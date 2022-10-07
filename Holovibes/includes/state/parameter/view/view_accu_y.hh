#pragma once

#include "custom_parameter.hh"
#include "view_struct.hh"

namespace holovibes
{

class View_Accu_YParam : public ICustomParameter<View_Accu_Y>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = View_XY{};

  public:
    View_Accu_YParam()
        : ICustomParameter<View_Accu_Y>(DEFAULT_VALUE)
    {
    }

    View_Accu_YParam(TransfertType value)
        : ICustomParameter<View_Accu_Y>(value)
    {
    }

  public:
    static const char* static_key() { return "View_Accu_Y"; }
    const char* get_key() const override { return View_Accu_YParam::static_key(); }
};

} // namespace holovibes
