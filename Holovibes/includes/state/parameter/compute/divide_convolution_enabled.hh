#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class DivideConvolutionEnable : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    DivideConvolutionEnable()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    DivideConvolutionEnable(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "divide_convolution_enabled"; }
    const char* get_key() const override { return DivideConvolutionEnable::static_key(); }
};

} // namespace holovibes
