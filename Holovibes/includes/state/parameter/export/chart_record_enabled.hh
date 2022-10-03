#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class ChartDisplayEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    ChartDisplayEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    ChartDisplayEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "chart_display_enabled"; }
    const char* get_key() const override { return ChartDisplayEnabled::static_key(); }
};

} // namespace holovibes
