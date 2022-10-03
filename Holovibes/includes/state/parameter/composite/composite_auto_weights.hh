#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class CompositeKind : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    CompositeKind()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    CompositeKind(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "composite_auto_weights"; }
    const char* get_key() const override { return CompositeKind::static_key(); }
};

} // namespace holovibes
