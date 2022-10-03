#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class Filter2dEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    Filter2dEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    Filter2dEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "filter2d_enabled"; }
    const char* get_key() const override { return Filter2dEnabled::static_key(); }
};

} // namespace holovibes
