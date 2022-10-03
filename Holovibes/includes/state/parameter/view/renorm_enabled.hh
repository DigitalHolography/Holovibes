#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class RenormEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    RenormEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    RenormEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "renorm_enabled"; }
    const char* get_key() const override { return RenormEnabled::static_key(); }
};

} // namespace holovibes
