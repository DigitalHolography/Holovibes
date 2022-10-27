#pragma once

#include "pipe_request_on_sync.hh"
#include "export_struct.hh"

namespace holovibes
{
class ExportPipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

    template <>
    void
    operator()<FrameRecordMode>(const FrameRecordStruct& new_value, const FrameRecordStruct& old_value, Pipe& pipe);

    template <>
    void operator()<ChartRecord>(const ChartRecordStruct& new_value, const ChartRecordStruct& old_value, Pipe& pipe);
};
} // namespace holovibes
