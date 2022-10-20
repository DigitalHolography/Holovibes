#pragma once

#include "API_detail.hh"
#include "pipe_API.hh"
#include "contrast_API.hh"

namespace holovibes::api
{

inline void set_x_cuts(uint value)
{
    auto fd = api::get_gpu_input_queue().get_fd();
    if (value < fd.width)
    {
        api::detail::change_value<ViewAccuX>()->set_cuts(value);
        pipe_refresh();
    }
}

inline void set_y_cuts(uint value)
{
    auto fd = api::get_gpu_input_queue().get_fd();
    if (value < fd.height)
    {
        api::detail::change_value<ViewAccuY>()->set_cuts(value);
        pipe_refresh();
    }
}

inline void set_auto_contrast_cuts()
{
    get_compute_pipe().request_autocontrast(WindowKind::XZview);
    get_compute_pipe().request_autocontrast(WindowKind::YZview);
}

bool set_3d_cuts_view(uint time_transformation_size);
void cancel_time_transformation_cuts(std::function<void()> callback);

} // namespace holovibes::api
