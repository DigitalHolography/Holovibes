#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class ComputePipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

  public:
    template <>
    void operator()<LastImageType>(const std::string&, const std::string&, Pipe& pipe)
    {
        request_pipe_refresh();
    }
};
} // namespace holovibes
