#pragma once

#include "pipe.hh"
#include "logger.hh"
#include "micro_cache.hh"

namespace holovibes
{
class CompositePipeRequest
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

};
} // namespace holovibes
