#pragma once

#include "custom_parameter.hh"
#include "view_struct.hh"

namespace holovibes
{

class View_Accu_QParam : public ICustomParameter<View_Accu_Q>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = View_PQ{};

  public:
    View_Accu_QParam()
        : ICustomParameter<View_Accu_Q>(DEFAULT_VALUE)
    {
    }

    View_Accu_QParam(TransfertType value)
        : ICustomParameter<View_Accu_Q>(value)
    {
    }

  public:
    static const char* static_key() { return "View_Accu_Q"; }
    const char* get_key() const override { return View_Accu_QParam::static_key(); }
};

} // namespace holovibes
