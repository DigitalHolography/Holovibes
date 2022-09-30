#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class BatchSize : public IUIntParameter
{
  public:
    BatchSize()
        : IUIntParameter()
    {
    }

    BatchSize(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "batch_size"; }
    const char* get_key() override { return BatchSize::static_key(); }
};

} // namespace holovibes
