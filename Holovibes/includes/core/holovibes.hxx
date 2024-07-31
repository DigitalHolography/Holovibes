#pragma once

#include "holovibes.hh"

namespace holovibes
{
inline std::shared_ptr<BatchInputQueue> Holovibes::get_input_queue() { return input_queue_.load(); }

inline std::shared_ptr<Queue> Holovibes::get_gpu_output_queue() { return gpu_output_queue_.load(); }

inline std::atomic<std::shared_ptr<Queue>> Holovibes::get_record_queue() { return record_queue_.load(); }

inline std::shared_ptr<Pipe> Holovibes::get_compute_pipe()
{
    auto loaded = compute_pipe_.load();
    if (!loaded)
    {
        throw std::runtime_error("Pipe is not initialized");
    }
    return loaded;
}

inline std::shared_ptr<Pipe> Holovibes::get_compute_pipe_no_throw() { return compute_pipe_.load(); }

inline const char* Holovibes::get_camera_ini_name() const { return active_camera_->get_ini_name(); }

inline const Holovibes::CudaStreams& Holovibes::get_cuda_streams() const { return cuda_streams_; }
} // namespace holovibes
