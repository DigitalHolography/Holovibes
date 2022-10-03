#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class UnwrapHistorySize : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 1;

  public:
    UnwrapHistorySize()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    UnwrapHistorySize(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "unwrap_history_size"; }
    const char* get_key() const override { return UnwrapHistorySize::static_key(); }
};

} // namespace holovibes
