#pragma once

#include "enum_computation.hh"
#include "custom_parameter.hh"

namespace holovibes
{
class ComputeMode : public ICustomParameter<Computation>
{
  public:
    static constexpr ValueType DEFAULT_VALUE = Computation::Raw;

  public:
    ComputeMode()
        : ICustomParameter<Computation>(DEFAULT_VALUE)
    {
    }

    ComputeMode(TransfertType value)
        : ICustomParameter<Computation>(value)
    {
    }

  public:
    static const char* static_key() { return "compute_mode"; }
    const char* get_key() const override { return ComputeMode::static_key(); }
};

} // namespace holovibes
