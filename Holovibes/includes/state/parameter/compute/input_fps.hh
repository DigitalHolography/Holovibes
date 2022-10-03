#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class InputFps : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 60;

  public:
    InputFps()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    InputFps(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "input_fps"; }
    const char* get_key() const override { return InputFps::static_key(); }
};

} // namespace holovibes
