#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class RecordBufferSize : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 1024;

  public:
    RecordBufferSize()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    RecordBufferSize(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "record_buffer_size"; }
    const char* get_key() const override { return RecordBufferSize::static_key(); }
};

} // namespace holovibes
