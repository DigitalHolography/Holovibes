/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#pragma once

#include "holovibes.hh"

namespace holovibes
{
    inline std::shared_ptr<Queue> Holovibes::get_gpu_input_queue()
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

    inline ComputeDescriptor& Holovibes::get_cd()
    {
        return cd_;
    }

    inline InformationContainer& Holovibes::get_info_container()
    {
        return info_container_;
    }

    inline void Holovibes::set_cd(const ComputeDescriptor& cd)
    {
        cd_ = cd;
    }

    inline const char* Holovibes::get_camera_ini_path() const
    {
        return active_camera_->get_ini_path();
    }
}