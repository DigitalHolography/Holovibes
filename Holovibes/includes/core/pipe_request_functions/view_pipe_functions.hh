#pragma once

#include "pipe.hh"
#include "logger.hh"
#include "micro_cache.hh"

namespace holovibes
{
class ViewPipeRequest
{
  public:
    template <typename T>
    void operator()(typename T::ConstRefType, typename T::ConstRefType, Pipe& pipe)
    {
    }

    template <>
    void operator()<RawViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE RawViewEnabled");

        if (new_value == false)
            pipe.get_gpu_raw_view_queue().reset(nullptr);
        else
        {
            auto fd = pipe.get_gpu_input_queue().get_fd();
            pipe.get_gpu_raw_view_queue().reset(new Queue(fd, GSH::instance().get_value<OutputBufferSize>()));
        }
    }

    template <>
    void operator()<ChartDisplayEnabled>(bool new_value, bool old_value, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE ChartDisplayEnabled");

        if (new_value == false)
            chart_env_.chart_display_queue_.reset(nullptr);
        else
            chart_env_.chart_display_queue_.reset(new ConcurrentDeque<ChartPoint>());
    }

    template <>
    void operator()<Filter2DViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE Filter2DViewEnabled");

        if (new_value == false)
            gpu_filter2d_view_queue_.reset(nullptr);
        else
        {
            auto fd = gpu_output_queue_.get_fd();
            gpu_filter2d_view_queue_.reset(new Queue(fd, GSH::instance().get_value<OutputBufferSize>()));
        }
    }

    template <>
    void operator()<LensViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
    {
        LOG_DEBUG(compute_worker, "UPDATE LensViewEnabled");

        if (new_value == false)
            fourier_transforms_->get_lens_queue().reset(nullptr);
    }
};
} // namespace holovibes
