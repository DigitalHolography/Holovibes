#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class BatchSize : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 1;

  public:
    BatchSize()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    BatchSize(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "batch_size"; }
    const char* get_key() const override { return BatchSize::static_key(); }
};

} // namespace holovibes
