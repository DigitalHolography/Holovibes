#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class ConvolutionEnable : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    ConvolutionEnable()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    ConvolutionEnable(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "convolution_enable"; }
    const char* get_key() const override { return ConvolutionEnable::static_key(); }
};

} // namespace holovibes
