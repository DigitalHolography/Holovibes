#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class CutsViewEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    CutsViewEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    CutsViewEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "cuts_view_enabled"; }
    const char* get_key() const override { return CutsViewEnabled::static_key(); }
};

} // namespace holovibes
