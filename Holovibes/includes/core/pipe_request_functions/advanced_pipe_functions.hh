#pragma once

#include "micro_cache.hh"
#include "pipe.hh"
#include "logger.hh"

namespace holovibes
{
class AdvancedPipeRequest
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
