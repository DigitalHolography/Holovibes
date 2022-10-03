#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class InputBufferSize : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 512;

  public:
    InputBufferSize()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    InputBufferSize(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "input_buffer_size"; }
    const char* get_key() const override { return InputBufferSize::static_key(); }
};

} // namespace holovibes
