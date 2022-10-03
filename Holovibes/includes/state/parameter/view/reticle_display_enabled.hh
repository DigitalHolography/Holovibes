#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class ReticleDisplayEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    ReticleDisplayEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    ReticleDisplayEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "reticle_display_enabled"; }
    const char* get_key() const override { return ReticleDisplayEnabled::static_key(); }
};

} // namespace holovibes
