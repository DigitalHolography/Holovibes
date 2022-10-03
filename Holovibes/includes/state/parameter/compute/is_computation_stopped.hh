#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class IsComputationStopped : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = true;

  public:
    IsComputationStopped()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    IsComputationStopped(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "is_computation_stopped"; }
    const char* get_key() const override { return IsComputationStopped::static_key(); }
};

} // namespace holovibes
