#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class LensViewEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    LensViewEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    LensViewEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "lens_view_enabled"; }
    const char* get_key() const override { return LensViewEnabled::static_key(); }
};

} // namespace holovibes
