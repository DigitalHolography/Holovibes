#pragma once

#include "pipe_request_on_sync.hh"

namespace holovibes
{
class ViewPipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

    template <>
    void operator()<RawViewEnabled>(bool new_value, bool old_value, Pipe& pipe);

    template <>
    void operator()<ChartDisplayEnabled>(bool new_value, bool old_value, Pipe& pipe);

    template <>
    void operator()<Filter2DViewEnabled>(bool new_value, bool old_value, Pipe& pipe);

    template <>
    void operator()<LensViewEnabled>(bool new_value, bool old_value, Pipe& pipe);
};
} // namespace holovibes
