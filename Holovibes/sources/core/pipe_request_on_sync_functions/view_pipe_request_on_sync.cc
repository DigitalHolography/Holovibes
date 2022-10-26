#pragma once

#include "API.hh"

namespace holovibes
{
template <>
void ViewPipeRequestOnSync::operator()<RawViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE RawViewEnabled");

    if (new_value == false)
        pipe.get_gpu_raw_view_queue().reset(nullptr);
    else
    {
        auto fd = pipe.get_gpu_input_queue().get_fd();
        pipe.get_gpu_raw_view_queue().reset(new Queue(fd, GSH::instance().get_value<OutputBufferSize>()));
    }
}

template <>
void ViewPipeRequestOnSync::operator()<ChartDisplayEnabled>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE ChartDisplayEnabled");

    if (new_value == false)
        pipe.get_chart_env().chart_display_queue_.reset(nullptr);
    else
        pipe.get_chart_env().chart_display_queue_.reset(new ConcurrentDeque<ChartPoint>());
}

template <>
void ViewPipeRequestOnSync::operator()<Filter2DViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE Filter2DViewEnabled");

    if (new_value == false)
        pipe.get_gpu_filter2d_view_queue().reset(nullptr);
    else
    {
        auto fd = pipe.get_gpu_output_queue().get_fd();
        pipe.gpu_filter2d_view_queue().reset(new Queue(fd, GSH::instance().get_value<OutputBufferSize>()));
    }
}

template <>
void ViewPipeRequestOnSync::operator()<LensViewEnabled>(bool new_value, bool old_value, Pipe& pipe)
{
    LOG_TRACE(compute_worker, "UPDATE LensViewEnabled");

    if (new_value == false)
        pipe.get_fourier_transforms().get_lens_queue().reset(nullptr);
}
} // namespace holovibes
