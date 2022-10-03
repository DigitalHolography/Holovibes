#pragma once

#include "bool_parameter.hh"

namespace holovibes
{
class FrameRecordEnabled : public IBoolParameter
{
  public:
    static constexpr ValueType DEFAULT_VALUE = false;

  public:
    FrameRecordEnabled()
        : IBoolParameter(DEFAULT_VALUE)
    {
    }

    FrameRecordEnabled(TransfertType value)
        : IBoolParameter(value)
    {
    }

  public:
    static const char* static_key() { return "frame_record_enabled"; }
    const char* get_key() const override { return FrameRecordEnabled::static_key(); }
};

} // namespace holovibes
