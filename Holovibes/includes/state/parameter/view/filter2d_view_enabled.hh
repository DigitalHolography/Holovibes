#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class Filter2dViewEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    Filter2dViewEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    Filter2dViewEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "filter2d_view_enabled"; }
    const char* get_key() const override { return Filter2dViewEnabled::static_key(); }
};

} // namespace holovibes
