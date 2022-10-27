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
    void operator()<ViewXY>(const View_XYZ&, const View_XYZ&, Pipe& pipe);

    template <>
    void operator()<ViewXZ>(const View_XYZ&, const View_XYZ&, Pipe& pipe);

    template <>
    void operator()<ViewYZ>(const View_XYZ&, const View_XYZ&, Pipe& pipe);

  public:
    template <>
    void operator()<Filter2D>(const View_Window&, const View_Window&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuX>(const View_XY&, const View_XY&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuY>(const View_XY&, const View_XY&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuP>(const View_PQ&, const View_PQ&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuQ>(const View_PQ&, const View_PQ&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<FftShiftEnabled>(bool, bool, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ReticleDisplayEnabled>(bool, bool, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ReticleScale>(float, float, Pipe& pipe)
    {
        request_pipe_refresh();
    }
};
} // namespace holovibes
