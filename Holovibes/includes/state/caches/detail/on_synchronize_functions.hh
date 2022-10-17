#pragma once

#include "micro_cache.hh"
#include "logger.hh"

namespace holovibes
{

class OnSynchronizeFunctions
{
  public:
    template <typename T>
    void operator()(T& value)
    {
        value.set_has_been_synchronized(false);
    }

    template <typename MicroCache>
    void call_on_cache(MicroCache& params)
    {
        params.synchronize();
    }
};

} // namespace holovibes
