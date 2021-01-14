/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "holovibes.hh"

namespace holovibes
{
inline std::shared_ptr<BatchInputQueue> Holovibes::get_gpu_input_queue()
{
    return gpu_input_queue_.load();
}

inline std::shared_ptr<Queue> Holovibes::get_gpu_output_queue()
{
    return gpu_output_queue_.load();
}

inline std::shared_ptr<ICompute> Holovibes::get_compute_pipe()
{
    if (!compute_pipe_.load())
        throw std::runtime_error("Pipe is not initialized");

    return compute_pipe_.load();
}

inline ComputeDescriptor& Holovibes::get_cd() { return cd_; }

inline InformationContainer& Holovibes::get_info_container()
{
    return info_container_;
}

inline void Holovibes::set_cd(const ComputeDescriptor& cd) { cd_ = cd; }

inline const char* Holovibes::get_camera_ini_path() const
{
    return active_camera_->get_ini_path();
}

inline const Holovibes::CudaStreams& Holovibes::get_cuda_streams() const
{
    return cuda_streams_;
}
} // namespace holovibes