#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class ChartRecordEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    ChartRecordEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    ChartRecordEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "chart_record_enabled"; }
    const char* get_key() const override { return ChartRecordEnabled::static_key(); }
};

} // namespace holovibes
