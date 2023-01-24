#pragma once

#include "pipe_request_on_sync.hh"
#include "view_struct.hh"

namespace holovibes
{
class ViewPipeRequestOnSync : public PipeRequestOnSync
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, Pipe&)
    {
    }

    template <typename T>
    void on_sync(typename T::ConstRefType new_value, [[maybe_unused]] typename T::ConstRefType, Pipe& pipe)
    {
        operator()<T>(new_value, pipe);
    }

  public:
    template <>
    void operator()<RawViewEnabled>(bool new_value, Pipe& pipe);

    template <>
    void operator()<ChartDisplayEnabled>(bool new_value, Pipe& pipe);

    template <>
    void operator()<Filter2DViewEnabled>(bool new_value, Pipe& pipe);

    template <>
    void operator()<LensViewEnabled>(bool new_value, Pipe& pipe);

    template <>
    void on_sync<ViewXY>(const ViewXYZ&, const ViewXYZ&, Pipe& pipe);
    template <>
    void operator()<ViewXY>(const ViewXYZ&, Pipe& pipe);

    template <>
    void on_sync<ViewXZ>(const ViewXYZ&, const ViewXYZ&, Pipe& pipe);
    template <>
    void operator()<ViewXZ>(const ViewXYZ&, Pipe& pipe);

    template <>
    void on_sync<ViewYZ>(const ViewXYZ&, const ViewXYZ&, Pipe& pipe);
    template <>
    void operator()<ViewYZ>(const ViewXYZ&, Pipe& pipe);

  public:
    template <>
    void operator()<ViewFilter2D>(const ViewWindow&, Pipe& pipe)
    {
        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuX>(const ViewAccuXY&, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(ViewAccuX);

        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuY>(const ViewAccuXY&, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(ViewAccuY);

        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuP>(const ViewAccuPQ&, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(ViewAccuP);

        request_pipe_refresh();
    }

    template <>
    void operator()<ViewAccuQ>(const ViewAccuPQ&, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(ViewAccuQ);

        request_pipe_refresh();
    }

    template <>
    void operator()<FftShiftEnabled>(bool, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(FftShiftEnabled);

        request_pipe_refresh();
    }

    template <>
    void operator()<Reticle>(const ReticleStruct&, Pipe& pipe)
    {
        LOG_UPDATE_ON_SYNC(Reticle);

        request_pipe_refresh();
    }
};
} // namespace holovibes
