#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class AdvancedPipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

    template <>
    void operator()<OutputBufferSize>(uint new_value, uint old_value, Pipe& pipe)
    {
        // FIXME : Not used
        // pipe.get_gpu_output_queue().resize(new_value, stream_);
    }
};
} // namespace holovibes
