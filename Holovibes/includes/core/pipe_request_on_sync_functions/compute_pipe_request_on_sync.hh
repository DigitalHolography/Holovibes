#pragma once

#include "pipe_request_on_sync.hh"
#include "rendering_struct.hh"

namespace holovibes
{
class ComputePipeRequestOnSync : public PipeRequestOnSync
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
    void operator()<BatchSize>(int new_value, Pipe& pipe);

    template <>
    void operator()<TimeStride>(int new_value, Pipe& pipe);

    template <>
    void operator()<TimeTransformationSize>(uint new_value, Pipe& pipe);

    template <>
    void operator()<Convolution>(const ConvolutionStruct& new_value, Pipe& pipe);

    template <>
    void operator()<TimeTransformationCutsEnable>(bool new_value, Pipe& pipe);

  public:
    template <>
    void operator()<Filter2D>(const Filter2DStruct&, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(Filter2D);

        request_pipe_refresh();
    }

    template <>
    void operator()<Lambda>(float, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(Lambda);

        request_pipe_refresh();
    }

    template <>
    void operator()<ZDistance>(float, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(ZDistance);

        request_pipe_refresh();
    }

    template <>
    void operator()<Unwrap2DRequested>(bool, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(Unwrap2DRequested);

        request_pipe_refresh();
    }

    template <>
    void operator()<SpaceTransformation>(SpaceTransformationEnum, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(SpaceTransformation);

        request_pipe_refresh();
    }

    template <>
    void operator()<TimeTransformation>(TimeTransformationEnum, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(TimeTransformation);

        request_pipe_refresh();
    }
};
} // namespace holovibes
