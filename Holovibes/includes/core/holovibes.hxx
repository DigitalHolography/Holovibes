#pragma once

#include "holovibes.hh"

namespace holovibes
{
inline std::shared_ptr<BatchInputQueue> Holovibes::get_gpu_input_queue() { return gpu_input_queue_.load(); }

inline std::shared_ptr<Queue> Holovibes::get_gpu_output_queue() { return gpu_output_queue_.load(); }

inline std::shared_ptr<Pipe> Holovibes::get_compute_pipe()
{
    if (!compute_pipe_.load())
        throw std::runtime_error("Pipe is not initialized");

    return compute_pipe_.load();
}

inline std::shared_ptr<Pipe> Holovibes::get_compute_pipe_nothrow() { return compute_pipe_.load(); }

inline const char* Holovibes::get_camera_ini_name() const { return active_camera_->get_ini_name(); }

inline const Holovibes::CudaStreams& Holovibes::get_cuda_streams() const { return cuda_streams_; }
} // namespace holovibes
