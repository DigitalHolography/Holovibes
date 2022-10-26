#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class RequestPipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

    template <>
    void operator()<RequestClearImgAccu>(TriggerRequest new_value, TriggerRequest old_value, Pipe& pipe);
};
} // namespace holovibes
