#pragma once

#include "micro_cache_tmp.hh"
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

    template <typename MicroCacheTmp>
    void call_on_cache(MicroCacheTmp& params)
    {
        params.synchronize();
    }
};

} // namespace holovibes
