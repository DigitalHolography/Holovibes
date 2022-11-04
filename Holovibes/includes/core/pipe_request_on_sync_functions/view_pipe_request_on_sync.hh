#pragma once

#include "pipe_request_on_sync.hh"
#include "view_struct.hh"

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

    template <>
    void operator()<ViewXY>(const ViewXYZ&, const ViewXYZ&, Pipe& pipe);

    template <>
    void operator()<ViewXZ>(const ViewXYZ&, const ViewXYZ&, Pipe& pipe);

    template <>
    void operator()<ViewYZ>(const ViewXYZ&, const ViewXYZ&, Pipe& pipe);

  public:
    template <>
    void operator()<ViewFilter2D>(const ViewWindow&, const ViewWindow&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuX>(const ViewAccuXY&, const ViewAccuXY&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuY>(const ViewAccuXY&, const ViewAccuXY&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuP>(const ViewAccuPQ&, const ViewAccuPQ&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuQ>(const ViewAccuPQ&, const ViewAccuPQ&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<FftShiftEnabled>(bool, bool, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<Reticle>(const ReticleStruct&, const ReticleStruct&, Pipe& pipe)
    {
        request_pipe_refresh();
    }
};
} // namespace holovibes
