#pragma once

#include "pipe.hh"
#include "logger.hh"
#include "micro_cache_tmp.hh"
#include "on_synchronize_functions.hh"

namespace holovibes
{
class PipeRequestFunctions
{
  public:
    using BeforeMethods = OnSynchronizeFunctions;

  public:
    template <typename T>
    bool test(const T& value)
    {
        return value.get_has_been_synchronized();
    }
};
} // namespace holovibes
