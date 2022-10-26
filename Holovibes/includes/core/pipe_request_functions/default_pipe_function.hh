#pragma once

#include "micro_cache.hh"
#include "pipe.hh"
#include "logger.hh"

namespace holovibes
{
class DefaultPipeRequest
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }
};
} // namespace holovibes
