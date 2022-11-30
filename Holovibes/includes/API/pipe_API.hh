#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline Pipe& get_compute_pipe() { return *Holovibes::instance().get_compute_pipe(); };
inline BatchInputQueue& get_gpu_input_queue() { return *Holovibes::instance().get_gpu_input_queue(); };
inline Queue& get_gpu_output_queue() { return *Holovibes::instance().get_gpu_output_queue(); };

inline std::shared_ptr<Pipe>& get_compute_pipe_ptr() { return Holovibes::instance().get_compute_pipe(); };

inline std::shared_ptr<BatchInputQueue>& get_gpu_input_queue_ptr()
{
    return Holovibes::instance().get_gpu_input_queue();
};
inline std::shared_ptr<Queue>& get_gpu_output_queue_ptr() { return Holovibes::instance().get_gpu_output_queue(); };

} // namespace holovibes::api
