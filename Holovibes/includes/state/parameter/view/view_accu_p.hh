#pragma once

#include "custom_parameter.hh"
#include "view_struct.hh"

namespace holovibes
{

class View_Accu_PParam : public ICustomParameter<View_Accu_P>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = View_PQ{};

  public:
    View_Accu_PParam()
        : ICustomParameter<View_Accu_P>(DEFAULT_VALUE)
    {
    }

    View_Accu_PParam(TransfertType value)
        : ICustomParameter<View_Accu_P>(value)
    {
    }

  public:
    static const char* static_key() { return "View_Accu_P"; }
    const char* get_key() const override { return View_Accu_PParam::static_key(); }
};

} // namespace holovibes
