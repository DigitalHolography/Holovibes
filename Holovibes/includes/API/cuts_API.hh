#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

void set_x_cuts(uint value)
{
    auto& holo = Holovibes::instance();
    const auto& fd = holo.get_gpu_input_queue()->get_fd();
    if (value < fd.width)
    {
        api::detail::change_value<ViewX>().cuts = value;
        pipe_refresh();
    }
}

void set_y_cuts(uint value)
{
    auto& holo = Holovibes::instance();
    const auto& fd = holo.get_gpu_input_queue()->get_fd();
    if (value < fd.height)
    {
        api::detail::change_value<ViewY>().cuts = value;
        pipe_refresh();
    }
}

} // namespace holovibes::api
