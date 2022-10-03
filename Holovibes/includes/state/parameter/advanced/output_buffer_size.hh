#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class OutputBufferSize : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 256;

  public:
    OutputBufferSize()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    OutputBufferSize(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "output_buffer_size"; }
    const char* get_key() const override { return OutputBufferSize::static_key(); }
};

} // namespace holovibes
