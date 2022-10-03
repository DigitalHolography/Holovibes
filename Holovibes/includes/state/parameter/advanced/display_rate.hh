#pragma once

#include "float_parameter.hh"

namespace holovibes
{
class DisplayRate : public IFloatParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 30;

  public:
    DisplayRate()
        : IFloatParameter(DEFAULT_VALUE)
    {
    }

    DisplayRate(TransfertType value)
        : IFloatParameter(value)
    {
    }

  public:
    static const char* static_key() { return "display_rate"; }
    const char* get_key() const override { return DisplayRate::static_key(); }
};

} // namespace holovibes
