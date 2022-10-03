#pragma once

#include "float_parameter.hh"

namespace holovibes
{
class Lambda : public IFloatParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 852e-9f;

  public:
    Lambda()
        : IFloatParameter(DEFAULT_VALUE)
    {
    }

    Lambda(TransfertType value)
        : IFloatParameter(value)
    {
    }

  public:
    static const char* static_key() { return "lambda"; }
    const char* get_key() const override { return Lambda::static_key(); }
};

} // namespace holovibes
