#pragma once

#include "float_parameter.hh"

namespace holovibes
{
class ZDistance : public IFloatParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 1.50f;

  public:
    ZDistance()
        : IFloatParameter(DEFAULT_VALUE)
    {
    }

    ZDistance(TransfertType value)
        : IFloatParameter(value)
    {
    }

  public:
    static const char* static_key() { return "z_distance"; }
    const char* get_key() const override { return ZDistance::static_key(); }
};

} // namespace holovibes
