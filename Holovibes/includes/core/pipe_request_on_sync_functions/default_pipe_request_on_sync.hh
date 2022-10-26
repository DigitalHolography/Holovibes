#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class DefaultPipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }
};
} // namespace holovibes
