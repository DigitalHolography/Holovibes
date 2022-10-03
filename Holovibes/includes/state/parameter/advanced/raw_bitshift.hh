#pragma once

#include "int_parameter.hh"

namespace holovibes
{
class RawBitshift : public IIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 0;

  public:
    RawBitshift()
        : IIntParameter(DEFAULT_VALUE)
    {
    }

    RawBitshift(TransfertType value)
        : IIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "raw_bitshift"; }
    const char* get_key() const override { return RawBitshift::static_key(); }
};

} // namespace holovibes
