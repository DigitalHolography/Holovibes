#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class RawViewEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    RawViewEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    RawViewEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "raw_view_enabled"; }
    const char* get_key() const override { return RawViewEnabled::static_key(); }
};

} // namespace holovibes
