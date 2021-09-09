#include "holovibes.hh"
#include "icompute.hh"
#include "tools.hh"
#include <cuda_runtime.h>
#include <chrono>

namespace holovibes::worker
{
using MutexGuard = std::lock_guard<std::mutex>;

InformationWorker::InformationWorker(bool is_cli, InformationContainer& info)
    : Worker()
    , is_cli_(is_cli)
    , info_(info)
{
}

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

        auto waited_time =
            std::chrono::duration_cast<std::chrono::milliseconds>(tick - start)
                .count();
        if (waited_time >= 1000)
        {
            compute_fps(waited_time);

            std::shared_ptr<Queue> gpu_output_queue =
                Holovibes::instance().get_gpu_output_queue();
            std::shared_ptr<BatchInputQueue> gpu_input_queue =
                Holovibes::instance().get_gpu_input_queue();

            if (gpu_output_queue && gpu_input_queue)
            {
                output_frame_res = gpu_output_queue->get_fd().frame_res();
                input_frame_size = gpu_input_queue->get_fd().frame_size();
            }

            try
            {
                std::shared_ptr<ICompute> pipe =
                    Holovibes::instance().get_compute_pipe();
                std::unique_ptr<Queue>& gpu_frame_record_queue =
                    pipe->get_frame_record_queue();

                if (gpu_frame_record_queue)
                    record_frame_size =
                        gpu_frame_record_queue->get_fd().frame_size();
            }
            catch (const std::exception&)
            {
            }

            compute_throughput(cd,
                               output_frame_res,
                               input_frame_size,
                               record_frame_size);

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
        std::atomic<unsigned int>* input_fps_ref =
            info_.fps_map_.at(InformationContainer::FpsType::INPUT_FPS);
        input_fps_ = std::round(input_fps_ref->load() * (1000.f / waited_time));
        input_fps_ref->store(0);
    }

    if (info_.fps_map_.contains(InformationContainer::FpsType::OUTPUT_FPS))
    {
        std::atomic<unsigned int>* output_fps_ref =
            info_.fps_map_.at(InformationContainer::FpsType::OUTPUT_FPS);
        output_fps_ =
            std::round(output_fps_ref->load() * (1000.f / waited_time));
        output_fps_ref->store(0);
    }

    if (info_.fps_map_.contains(InformationContainer::FpsType::SAVING_FPS))
    {
        std::atomic<unsigned int>* saving_fps_ref =
            info_.fps_map_.at(InformationContainer::FpsType::SAVING_FPS);
        saving_fps_ =
            std::round(saving_fps_ref->load() * (1000.f / waited_time));
        saving_fps_ref->store(0);
    }
}

void InformationWorker::compute_throughput(ComputeDescriptor& cd,
                                           size_t output_frame_res,
                                           size_t input_frame_size,
                                           size_t record_frame_size)
{
    input_throughput_ = input_fps_ * input_frame_size;
    output_throughput_ =
        output_fps_ * output_frame_res * cd.time_transformation_size;
    saving_throughput_ = saving_fps_ * record_frame_size;
}

static std::string format_throughput(size_t throughput, const std::string& unit)
{
    float throughput_ = throughput / (throughput > 1e9 ? 1e9 : 1e6);
    std::string unit_ = (throughput > 1e9 ? " G" : " M") + unit;
    std::stringstream ss;
    ss << std::fixed << std::setprecision(2) << throughput_ << unit_;
    return ss.str();
}

void InformationWorker::display_gui_information()
{
    MutexGuard m_guard(info_.mutex_);

    std::string to_display;
    to_display.reserve(512);

    for (auto const& [key, value] : info_.indication_map_)
        to_display +=
            info_.indication_type_to_string_.at(key) + ":\n  " + value + "\n";

    for (auto const& [key, value] : info_.queue_size_map_)
    {
        to_display += info_.queue_type_to_string_.at(key) + ":\n  ";
        to_display += std::to_string(value.first->load()) + "/" +
                      std::to_string(value.second->load()) + "\n";
    }

    if (info_.fps_map_.contains(InformationContainer::FpsType::INPUT_FPS))
    {
        to_display += info_.fps_type_to_string_.at(
                          InformationContainer::FpsType::INPUT_FPS) +
                      ":\n  " + std::to_string(input_fps_) + "\n";
    }

    if (info_.fps_map_.contains(InformationContainer::FpsType::OUTPUT_FPS))
    {
        to_display += info_.fps_type_to_string_.at(
                          InformationContainer::FpsType::OUTPUT_FPS) +
                      ":\n  " + std::to_string(output_fps_) + "\n";
    }

    if (info_.fps_map_.contains(InformationContainer::FpsType::SAVING_FPS))
    {
        to_display += info_.fps_type_to_string_.at(
                          InformationContainer::FpsType::SAVING_FPS) +
                      ":\n  " + std::to_string(saving_fps_) + "\n";
    }

    if (info_.fps_map_.contains(InformationContainer::FpsType::OUTPUT_FPS))
    {
        to_display += "Input Throughput\n  " +
                      format_throughput(input_throughput_, "B/s") + "\n";
        to_display += "Output Throughput\n  " +
                      format_throughput(output_throughput_, "Voxels/s") + "\n";
    }

    if (info_.fps_map_.contains(InformationContainer::FpsType::SAVING_FPS))
    {
        to_display += "Saving Throughput\n  " +
                      format_throughput(saving_throughput_, "B/s") + "\n";
    }

    size_t free, total;
    cudaMemGetInfo(&free, &total);
    to_display += "GPU memory:\n" + std::string("  ") +
                  engineering_notation(free, 3) + "B free,\n" +
                  std::string("  ") + engineering_notation(total, 3) +
                  "B total";

    info_.display_info_text_function_(to_display);

    for (auto const& [key, value] : info_.progress_index_map_)
        info_.update_progress_function_(key,
                                        value.first->load(),
                                        value.second->load());
}
} // namespace holovibes::worker
