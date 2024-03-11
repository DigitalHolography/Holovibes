#include "holovibes.hh"
#include "icompute.hh"
#include "tools.hh"
#include "chrono.hh"
#include <cuda_runtime.h>
#include <chrono>
#include "global_state_holder.hh"
#include <nvml.h>

namespace holovibes::worker
{
using MutexGuard = std::lock_guard<std::mutex>;

const std::unordered_map<IndicationType, std::string> InformationWorker::indication_type_to_string_ = {
    {IndicationType::IMG_SOURCE, "Image Source"},
    {IndicationType::INPUT_FORMAT, "Input Format"},
    {IndicationType::OUTPUT_FORMAT, "Output Format"}};

const std::unordered_map<FpsType, std::string> InformationWorker::fps_type_to_string_ = {
    {FpsType::INPUT_FPS, "Input FPS"},
    {FpsType::OUTPUT_FPS, "Output FPS"},
    {FpsType::SAVING_FPS, "Saving FPS"},
};

const std::unordered_map<QueueType, std::string> InformationWorker::queue_type_to_string_ = {
    {QueueType::INPUT_QUEUE, "Input Queue"},
    {QueueType::OUTPUT_QUEUE, "Output Queue"},
    {QueueType::RECORD_QUEUE, "Record Queue"},
};

void InformationWorker::run()
{
    std::shared_ptr<ICompute> pipe;
    unsigned int output_frame_res = 0;
    unsigned int input_frame_size = 0;
    unsigned int record_frame_size = 0;

    // Init start
    Chrono chrono;

    while (!stop_requested_)
    {
        chrono.stop();

        auto waited_time = chrono.get_milliseconds();
        if (waited_time >= 1000)
        {
            compute_fps(waited_time);

            std::shared_ptr<Queue> gpu_output_queue = Holovibes::instance().get_gpu_output_queue();
            std::shared_ptr<BatchInputQueue> gpu_input_queue = Holovibes::instance().get_gpu_input_queue();

            if (gpu_output_queue && gpu_input_queue)
            {
                output_frame_res = gpu_output_queue->get_fd().get_frame_res();
                input_frame_size = gpu_input_queue->get_fd().get_frame_size();
            }

            std::shared_ptr<ICompute> pipe = Holovibes::instance().get_compute_pipe_nothrow();
            if (pipe != nullptr)
            {
                std::unique_ptr<Queue>& frame_record_queue = pipe->get_frame_record_queue();
                if (frame_record_queue)
                    record_frame_size = frame_record_queue->get_fd().get_frame_size();
            }
            else
            {
                record_frame_size = 0;
            }

            compute_throughput(output_frame_res, input_frame_size, record_frame_size);

            chrono.start();
        }

        display_gui_information();

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
}

void InformationWorker::compute_fps(const long long waited_time)
{
    auto& fps_map = GSH::fast_updates_map<FpsType>;
    FastUpdatesHolder<FpsType>::const_iterator it;
    if ((it = fps_map.find(FpsType::INPUT_FPS)) != fps_map.end())
        input_fps_ = it->second->load();

    if ((it = fps_map.find(FpsType::OUTPUT_FPS)) != fps_map.end())
    {
        output_fps_ = std::round(it->second->load() * (1000.f / waited_time));
        it->second->store(0); // TODO Remove
    }

    if ((it = fps_map.find(FpsType::SAVING_FPS)) != fps_map.end())
    {
        saving_fps_ = std::round(it->second->load() * (1000.f / waited_time));
        it->second->store(0); // TODO Remove
    }
}

void InformationWorker::compute_throughput(size_t output_frame_res, size_t input_frame_size, size_t record_frame_size)
{
    input_throughput_ = input_fps_ * input_frame_size;
    output_throughput_ = output_fps_ * output_frame_res * setting<settings::TimeTransformationSize>();
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

std::string gpu_load()
{
    std::stringstream ss;
    ss << "GPU load: \n  ";
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t gpuLoad;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        return ss.str();
    }

    // Get the device handle (assuming only one GPU is present)
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        nvmlShutdown();
        return ss.str();
    }

    // Query GPU load
    result = nvmlDeviceGetUtilizationRates(device, &gpuLoad);
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        nvmlShutdown();
        return ss.str();
    }

    // Print GPU load
    ss << gpuLoad.gpu << "%";

    // Shutdown NVML
    nvmlShutdown();

    return ss.str();
}

std::string gpu_memory_load()
{
    std::stringstream ss;
    ss << "GPU memory controller load: \n  ";
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t gpuLoad;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        return ss.str();
    }

    // Get the device handle (assuming only one GPU is present)
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        nvmlShutdown();
        return ss.str();
    }

    // Query GPU load
    result = nvmlDeviceGetUtilizationRates(device, &gpuLoad);
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        nvmlShutdown();
        return ss.str();
    }

    // Print GPU load
    ss << gpuLoad.memory << "%";

    // Shutdown NVML
    nvmlShutdown();

    return ss.str();
}

void InformationWorker::display_gui_information()
{
    std::string str;
    str.reserve(512);
    std::stringstream to_display(str);
    auto& fps_map = GSH::fast_updates_map<FpsType>;

    for (auto const& [key, value] : GSH::fast_updates_map<IndicationType>)
        to_display << indication_type_to_string_.at(key) << ":\n  " << *value << "\n";

    for (auto const& [key, value] : GSH::fast_updates_map<QueueType>)
    {
        if (key == QueueType::UNDEFINED)
            continue;

        to_display << queue_type_to_string_.at(key) << ":\n  ";
        to_display << value->first.load() << "/" << value->second.load() << "\n";
    }

    if (fps_map.contains(FpsType::INPUT_FPS))
    {
        to_display << fps_type_to_string_.at(FpsType::INPUT_FPS) << ":\n  " << input_fps_ << "\n";
    }

    if (fps_map.contains(FpsType::OUTPUT_FPS))
    {
        to_display << fps_type_to_string_.at(FpsType::OUTPUT_FPS) << ":\n  " << output_fps_ << "\n";
    }

    if (fps_map.contains(FpsType::SAVING_FPS))
    {
        to_display << fps_type_to_string_.at(FpsType::SAVING_FPS) << ":\n  " << saving_fps_ << "\n";
    }

    if (fps_map.contains(FpsType::OUTPUT_FPS))
    {
        to_display << "Input Throughput\n  " << format_throughput(input_throughput_, "B/s") << "\n";
        to_display << "Output Throughput\n  " << format_throughput(output_throughput_, "Voxels/s") << "\n";
    }

    if (fps_map.contains(FpsType::SAVING_FPS))
    {
        to_display << "Saving Throughput\n  " << format_throughput(saving_throughput_, "B/s") << "\n";
    }

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    to_display << "GPU memory:\n"
               << std::string("  ") << engineering_notation(free, 3) << "B free,\n"
               << "  " << engineering_notation(total, 3) + "B total\n";

    to_display << gpu_load() << '\n';
    to_display << gpu_memory_load() << '\n';

    // #TODO change this being called every frame to only being called to update the value if needed
    to_display << "\nz boundary: " << Holovibes::instance().get_boundary() << "m\n";

    display_info_text_function_(to_display.str());

    for (auto const& [key, value] : GSH::fast_updates_map<ProgressType>)
        update_progress_function_(key, value->first.load(), value->second.load());
}
} // namespace holovibes::worker
