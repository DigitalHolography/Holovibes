#pragma once

#include "float_parameter.hh"

namespace holovibes
{
class ReticleScale : public IFloatParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 30;

  public:
    ReticleScale()
        : IFloatParameter(DEFAULT_VALUE)
    {
    }

    ReticleScale(TransfertType value)
        : IFloatParameter(value)
    {
    }

  public:
    static const char* static_key() { return "reticle_rate"; }
    const char* get_key() const override { return ReticleScale::static_key(); }
};

} // namespace holovibes
