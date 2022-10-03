#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class RenormConstant : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 5;

  public:
    RenormConstant()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    RenormConstant(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "renorm_constant"; }
    const char* get_key() const override { return RenormConstant::static_key(); }
};

} // namespace holovibes
