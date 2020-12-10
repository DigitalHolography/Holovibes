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

#include "holovibes.hh"
#include "icompute.hh"
#include "tools.hh"
#include <cuda_runtime.h>
#include <chrono>

namespace holovibes::worker
{
    using MutexGuard = std::lock_guard<std::mutex>;

    InformationWorker::InformationWorker(bool is_cli, InformationContainer& info):
        Worker(),
        is_cli_(is_cli),
        info_(info)
    {}

    void InformationWorker::run()
    {
        ComputeDescriptor& cd = Holovibes::instance().get_cd();
        unsigned int output_frame_res = 0;
        unsigned int input_frame_size = 0;
        unsigned int record_frame_size = 0;

        // Init start
        auto start = std::chrono::high_resolution_clock::now();

        while (!stop_requested_)
        {
            auto tick = std::chrono::high_resolution_clock::now();

            auto waited_time = std::chrono::duration_cast<std::chrono::milliseconds>(tick - start).count();
            if (waited_time >= 1000)
            {
                compute_fps(waited_time);

                std::shared_ptr<Queue> gpu_output_queue = Holovibes::instance().get_gpu_output_queue();
                std::shared_ptr<Queue> gpu_input_queue = Holovibes::instance().get_gpu_input_queue();

                if (gpu_output_queue && gpu_input_queue)
                {
                    output_frame_res = gpu_output_queue->get_fd().frame_res();
                    input_frame_size = gpu_input_queue->get_fd().frame_size();
                }

                try
                {
                    std::shared_ptr<ICompute> pipe = Holovibes::instance().get_compute_pipe();
                    std::unique_ptr<Queue>& gpu_frame_record_queue = pipe->get_frame_record_queue();

                    if (gpu_frame_record_queue)
                        record_frame_size = gpu_frame_record_queue->get_fd().frame_size();
                }
                catch (const std::exception&)
                {}

                compute_throughput(cd, output_frame_res, input_frame_size, record_frame_size);

                start = std::chrono::high_resolution_clock::now();
            }

            if (!is_cli_)
                display_gui_information();

            std::this_thread::sleep_for(std::chrono::milliseconds(50));
        }
    }

    void InformationWorker::compute_fps(const long long waited_time)
    {
        if (info_.fps_map_.contains(InformationContainer::FpsType::INPUT_FPS))
        {
            std::atomic<unsigned int>* input_fps_ref = info_.fps_map_.at(InformationContainer::FpsType::INPUT_FPS);
            input_fps_ = std::round(input_fps_ref->load() * (1000.f / waited_time));
            input_fps_ref->store(0);
        }

        if (info_.fps_map_.contains(InformationContainer::FpsType::OUTPUT_FPS))
        {
            std::atomic<unsigned int>* output_fps_ref = info_.fps_map_.at(InformationContainer::FpsType::OUTPUT_FPS);
            output_fps_ = std::round(output_fps_ref->load() * (1000.f / waited_time));
            output_fps_ref->store(0);
        }

        if (info_.fps_map_.contains(InformationContainer::FpsType::SAVING_FPS))
        {
            std::atomic<unsigned int>* saving_fps_ref = info_.fps_map_.at(InformationContainer::FpsType::SAVING_FPS);
            saving_fps_ = std::round(saving_fps_ref->load() * (1000.f / waited_time));
            saving_fps_ref->store(0);
        }
    }

    void InformationWorker::compute_throughput(ComputeDescriptor& cd,
        unsigned int output_frame_res, unsigned int input_frame_size, unsigned int record_frame_size)
    {
        input_throughput_ = std::round(input_fps_ * input_frame_size / 1e6f);
        output_throughput_ = std::round(output_fps_ * output_frame_res * cd.time_transformation_size / 1e6f);
        saving_throughput_ = std::round(saving_fps_ * record_frame_size / 1e6f);
    }

    void InformationWorker::display_gui_information()
    {
        MutexGuard m_guard(info_.mutex_);

        std::string to_display;

        for (auto const& [key, value] : info_.indication_map_)
            to_display += info_.indication_type_to_string_.at(key) + ":\n  " + value + "\n";

        for (auto const& [key, value] : info_.queue_size_map_)
        {
            to_display += info_.queue_type_to_string_.at(key) + ":\n  ";
            to_display += std::to_string(value.first->load()) + "/" + std::to_string(value.second->load()) + "\n";
        }

        if (info_.fps_map_.contains(InformationContainer::FpsType::INPUT_FPS))
        {
            to_display += info_.fps_type_to_string_.at(InformationContainer::FpsType::INPUT_FPS) + ":\n  "
                        + std::to_string(input_fps_) + "\n";

        }

        if (info_.fps_map_.contains(InformationContainer::FpsType::OUTPUT_FPS))
        {
            to_display += info_.fps_type_to_string_.at(InformationContainer::FpsType::OUTPUT_FPS) + ":\n  "
                        + std::to_string(output_fps_) + "\n";
        }

        if (info_.fps_map_.contains(InformationContainer::FpsType::SAVING_FPS))
        {
            to_display += info_.fps_type_to_string_.at(InformationContainer::FpsType::SAVING_FPS) + ":\n  "
                        + std::to_string(saving_fps_) + "\n";
        }

        if (info_.fps_map_.contains(InformationContainer::FpsType::OUTPUT_FPS))
        {
            // if output fps do not exist, the input and output throughputs are 0
            to_display += "Input Throughput:\n  " + std::to_string(input_throughput_) + "MB/s\n";
            to_display += "Output Throughput:\n  " + std::to_string(output_throughput_) + "MVoxel/s\n";
        }

        if (info_.fps_map_.contains(InformationContainer::FpsType::SAVING_FPS))
            to_display += "Saving Throughput:\n  " + std::to_string(saving_throughput_) + "MB/s\n";

        size_t free, total;
        cudaMemGetInfo(&free, &total);
        to_display += "GPU memory:\n"
                    + std::string("  ") + engineering_notation(free, 3) + "B free,\n"
                    + std::string("  ") + engineering_notation(total, 3) + "B total";

        info_.display_info_text_function_(to_display);

        for (auto const& [key, value] : info_.progress_index_map_)
            info_.update_progress_function_(key, value.first->load(), value.second->load());
    }
} // namespace holovibes::worker