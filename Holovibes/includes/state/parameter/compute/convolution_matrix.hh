#pragma once

#include "custom_parameter.hh"
#include <vector>

namespace holovibes
{
class ConvolutionMatrice : public ICustomParameter<std::vector<float>>
{
  public:
    static const ValueType DEFAULT_VALUE;

  public:
    ConvolutionMatrice()
        : ICustomParameter<std::vector<float>>(DEFAULT_VALUE)
    {
    }

    ConvolutionMatrice(TransfertType value)
        : ICustomParameter<std::vector<float>>(value)
    {
    }

  public:
    static const char* static_key() { return "convolution_matrix"; }
    const char* get_key() const override { return ConvolutionMatrice::static_key(); }
};

} // namespace holovibes
