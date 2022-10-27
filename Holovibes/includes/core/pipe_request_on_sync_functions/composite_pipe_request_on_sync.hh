#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class CompositePipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

  public:
    template <>
    void operator()<CompositeRGBParam>(const CompositeRGB&, const CompositeRGB&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<CompositeHSVParam>(const CompositeHSV&, const CompositeHSV&, Pipe& pipe)
    {
        request_pipe_refresh();
    }
};
} // namespace holovibes
