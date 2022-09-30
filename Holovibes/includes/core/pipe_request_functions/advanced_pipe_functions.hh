#pragma once

#include "detail.hh"

namespace holovibes
{
class AdvancedPipeRequest : public PipeRequestFunctions
{
  public:
    template <typename T>
    void operator()(const T&, Pipe& pipe)
    {
    }
};
} // namespace holovibes
