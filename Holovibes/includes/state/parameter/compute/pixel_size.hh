#pragma once

#include "float_parameter.hh"

namespace holovibes
{
class PixelSize : public IFloatParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 12.0f;

  public:
    PixelSize()
        : IFloatParameter(DEFAULT_VALUE)
    {
    }

    PixelSize(TransfertType value)
        : IFloatParameter(value)
    {
    }

  public:
    static const char* static_key() { return "pixel_size"; }
    const char* get_key() const override { return PixelSize::static_key(); }
};

} // namespace holovibes
