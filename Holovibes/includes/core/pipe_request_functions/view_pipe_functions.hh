#pragma once

#include "pipe.hh"
#include "logger.hh"
#include "micro_cache.hh"

namespace holovibes
{
class ViewPipeRequest
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

    template <>
    void operator()<RawViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE RawViewEnabled");

        if (new_value == false)
            pipe.get_gpu_raw_view_queue().reset(nullptr);
        else
        {
            auto fd = pipe.get_gpu_input_queue().get_fd();
            pipe.get_gpu_raw_view_queue().reset(new Queue(fd, GSH::instance().get_value<OutputBufferSize>()));
        }
    }
};
} // namespace holovibes
