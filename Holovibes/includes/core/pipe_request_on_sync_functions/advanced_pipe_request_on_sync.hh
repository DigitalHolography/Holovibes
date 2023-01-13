#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class AdvancedPipeRequestOnSync : public PipeRequestOnSync
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
    void operator()<InputBufferSize>(uint new_value, Pipe& pipe);
    template <>
    void operator()<OutputBufferSize>(uint new_value, Pipe& pipe);
    template <>
    void operator()<TimeTransformationCutsBufferSize>(uint new_value, Pipe& pipe);
};

} // namespace holovibes
