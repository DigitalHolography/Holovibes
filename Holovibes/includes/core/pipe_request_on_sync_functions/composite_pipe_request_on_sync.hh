#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class CompositePipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, Pipe&)
    {
    }

    template <typename T>
    void on_sync(typename T::ConstRefType new_value, [[maybe_unused]] typename T::ConstRefType, Pipe& pipe)
    {
        operator()<T>(new_value, pipe);
    }

  public:
    template <>
    void operator()<CompositeRGB>(const CompositeRGBStruct&, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(CompositeRGB);

        request_pipe_refresh();
    }

    template <>
    void operator()<CompositeHSV>(const CompositeHSVStruct&, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(CompositeHSV);

        request_pipe_refresh();
    }
};
} // namespace holovibes
