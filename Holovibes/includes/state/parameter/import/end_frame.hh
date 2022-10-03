#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class EndFrame : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 0;

  public:
    EndFrame()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    EndFrame(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "end_frame"; }
    const char* get_key() const override { return EndFrame::static_key(); }
};

} // namespace holovibes
