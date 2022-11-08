#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class ImportPipeRequestOnSync : public PipeRequestOnSync
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
    void operator()<LastImageType>(const std::string&, Pipe& pipe)
    {
        LOG_UPDATE_PIPE(LastImageType);

        request_pipe_refresh();
    }
};
} // namespace holovibes
