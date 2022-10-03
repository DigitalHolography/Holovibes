#pragma once

#include "uint_parameter.hh"

namespace holovibes
{
class StartFrame : public IUIntParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = 0;

  public:
    StartFrame()
        : IUIntParameter(DEFAULT_VALUE)
    {
    }

    StartFrame(TransfertType value)
        : IUIntParameter(value)
    {
    }

  public:
    static const char* static_key() { return "start_frame"; }
    const char* get_key() const override { return StartFrame::static_key(); }
};

} // namespace holovibes
