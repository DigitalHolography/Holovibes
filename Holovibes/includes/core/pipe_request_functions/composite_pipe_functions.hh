#pragma once

#include "detail.hh"
#include "pipe.hh"

namespace holovibes
{
class CompositePipeRequest : public PipeRequestFunctions
{
  public:
    template <typename T>
    void operator()(const T&, Pipe& pipe)
    {
    }
};
} // namespace holovibes
