#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class TimeStride : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 1;

  public:
    TimeStride()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    TimeStride(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "time_stride"; }
    const char* get_key() const override { return TimeStride::static_key(); }
};

} // namespace holovibes
