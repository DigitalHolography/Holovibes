#include <fstream>
#include "holovibes.hh"
#include "icompute.hh"
#include "tools.hh"
#include "chrono.hh"
#include <cuda_runtime.h>
#include <chrono>
#include "global_state_holder.hh"
#include <nvml.h>
#include "logger.hh"
#include "spdlog/spdlog.h"

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

    auto benchmark_mode = Holovibes::instance().get_setting<holovibes::settings::BenchmarkMode>().value;
    std::ofstream benchmark_file;
    bool info_found = false;

    if (benchmark_mode)
    {
        LOG_INFO("Benchmark mode active");
        std::string benchmark_file_path = settings::benchmark_dirpath + "/benchmark_NOW.csv";
        benchmark_file.open(benchmark_file_path);
        if (!benchmark_file.is_open())
            LOG_ERROR("Could not open benchmark file at " + benchmark_file_path + ", you may need to create the folder");
    }

    while (!stop_requested_)
    {
        chrono.stop();

        auto waited_time = chrono.get_milliseconds();
        if (waited_time >= 1000)
        {
            compute_fps(waited_time);

            std::shared_ptr<Queue> gpu_output_queue = Holovibes::instance().get_gpu_output_queue();
            std::shared_ptr<BatchInputQueue> input_queue = Holovibes::instance().get_input_queue();

            if (gpu_output_queue && input_queue)
            {
                output_frame_res = static_cast<unsigned int>(gpu_output_queue->get_fd().get_frame_res());
                input_frame_size = static_cast<unsigned int>(input_queue->get_fd().get_frame_size());
            }

            auto frame_record_queue = Holovibes::instance().get_record_queue().load();
            record_frame_size = 0;
            if (frame_record_queue)
                record_frame_size = static_cast<unsigned int>(frame_record_queue->get_fd().get_frame_size());
            // if (pipe != nullptr)
            // {
            //     std::unique_ptr<Queue>& frame_record_queue = pipe->get_frame_record_queue();
            //     if (frame_record_queue)
            //         record_frame_size = frame_record_queue->get_fd().get_frame_size();
            // }
            // else
            // {
            //     record_frame_size = 0;
            // }

            compute_throughput(output_frame_res, input_frame_size, record_frame_size);

            chrono.start();
        }

        display_gui_information();

        if (benchmark_mode)
        {
            if (!info_found)
            {
                if (!GSH::fast_updates_map<IndicationType>.empty())
                {    
                    // metadata
                    benchmark_file << "Version: 0";
                    for (auto const& [key, value] : GSH::fast_updates_map<IndicationType>)
                        benchmark_file << "," << indication_type_to_string_.at(key) << ": " << *value;
                    for (auto const& [key, value] : GSH::fast_updates_map<QueueType>)  //! FIXME causes a crash on start when camera pre-selected
                        benchmark_file << "," << (std::get<2>(*value).load() == Device::GPU ? "GPU " : "CPU ") << queue_type_to_string_.at(key) << " size: " << std::get<1>(*value).load();
                    benchmark_file << "\n";
                    // 11 headers
                    benchmark_file << "Input Queue,Output Queue,Record Queue,Input FPS,Output FPS,Input Throughput,Output Throughput,GPU memory free,GPU memory total,GPU load,GPU memory load, z_boundary\n";
                    info_found = true;
                }
            }
            else
                InformationWorker::write_information(benchmark_file);
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    if (benchmark_mode)
    {
        benchmark_file.close();
        // rename file
        std::string benchmark_file_path = settings::benchmark_dirpath + "/benchmark_" + Chrono::get_current_date_time() + ".csv";
        std::rename((settings::benchmark_dirpath + "/benchmark_NOW.csv").c_str(), benchmark_file_path.c_str());
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
    ss << "GPU load: <br/>  ";
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t gpuLoad;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS)
    {
        result = nvmlShutdown();
        nvmlShutdown();
        ss << "Could not load GPU usage";
        return ss.str();
    }

    // Get the device handle (assuming only one GPU is present)
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        result = nvmlShutdown();
        nvmlShutdown();
        return ss.str();
    }

    // Query GPU load
    result = nvmlDeviceGetUtilizationRates(device, &gpuLoad);
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        result = nvmlShutdown();
        nvmlShutdown();
        return ss.str();
    }

    // Print GPU load
    auto load = gpuLoad.gpu;
    ss << "<font color=";
    if (load < 80)
        ss << "white";
    else if (load < 90)
        ss << "orange";
    else
        ss << "red";
    ss << ">" << load << "</font>";
    ss << " %";

    // Shutdown NVML
    result = nvmlShutdown();
    nvmlShutdown();

    return ss.str();
}

std::string gpu_load_as_number()
{
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t gpuLoad;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS)
    {
        return "Could not load GPU usage";
    }

    // Get the device handle (assuming only one GPU is present)
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS)
    {
        result = nvmlShutdown();
        nvmlShutdown();
        return "Could not load GPU usage";
    }

    // Query GPU load
    result = nvmlDeviceGetUtilizationRates(device, &gpuLoad);
    if (result != NVML_SUCCESS)
    {
        result = nvmlShutdown();
        nvmlShutdown();
        return "Could not load GPU usage";
    }

    // Shutdown NVML
    result = nvmlShutdown();
    nvmlShutdown();

    return std::to_string(gpuLoad.gpu);
}

std::string gpu_memory_controller_load()
{
    std::stringstream ss;
    ss << "GPU memory controller load: <br/>  ";
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t gpuLoad;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        result = nvmlShutdown();
        nvmlShutdown();
        return ss.str();
    }

    // Get the device handle (assuming only one GPU is present)
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        result = nvmlShutdown();
        nvmlShutdown();
        return ss.str();
    }

    // Query GPU load
    result = nvmlDeviceGetUtilizationRates(device, &gpuLoad);
    if (result != NVML_SUCCESS)
    {
        ss << "Could not load GPU usage";
        result = nvmlShutdown();
        nvmlShutdown();
        return ss.str();
    }

    // Print GPU load
    auto load = gpuLoad.memory;
    ss << "<font color=";
    if (load < 80)
        ss << "white";
    else if (load < 90)
        ss << "orange";
    else
        ss << "red";
    ss << ">" << load << "</font>";
    ss << " %";

    // Shutdown NVML
    result = nvmlShutdown();
    nvmlShutdown();

    return ss.str();
}

std::string gpu_memory_controller_load_as_number()
{
    nvmlReturn_t result;
    nvmlDevice_t device;
    nvmlUtilization_t gpuLoad;

    // Initialize NVML
    result = nvmlInit();
    if (result != NVML_SUCCESS)
    {
        return "Could not load GPU usage";
    }

    // Get the device handle (assuming only one GPU is present)
    result = nvmlDeviceGetHandleByIndex(0, &device);
    if (result != NVML_SUCCESS)
    {
        result = nvmlShutdown();
        nvmlShutdown();
        return "Could not load GPU usage";
    }

    // Query GPU load
    result = nvmlDeviceGetUtilizationRates(device, &gpuLoad);
    if (result != NVML_SUCCESS)
    {
        result = nvmlShutdown();
        nvmlShutdown();
        return "Could not load GPU usage";
    }

    // Shutdown NVML
    result = nvmlShutdown();
    nvmlShutdown();

    return std::to_string(gpuLoad.memory);
}

std::string gpu_memory()
{
    std::stringstream ss;
    ss << "GPU memory: <br/>  ";
    size_t free, total;
    cudaMemGetInfo(&free, &total);
    // if free < 0.1 * total then red
    ss << "<font color=";
    if (free < 0.1 * total)
        ss << "red";
    else if (free < 0.25 * total)
        ss << "orange";
    else
        ss << "white";
    ss << ">" << engineering_notation(free, 3) << "B free,  " << "</font><br/>";
    ss << engineering_notation(total, 3) << "B total";

    return ss.str();
}

void InformationWorker::display_gui_information()
{
    std::string str;
    str.reserve(512);
    std::stringstream to_display(str);
    auto& fps_map = GSH::fast_updates_map<FpsType>;

    for (auto const& [key, value] : GSH::fast_updates_map<IndicationType>)
        to_display << indication_type_to_string_.at(key) << ":<br/>  " << *value << "<br/>";

    for (auto const& [key, value] : GSH::fast_updates_map<QueueType>)
    {
        if (key == QueueType::UNDEFINED)
            continue;
        auto currentLoad = std::get<0>(*value).load();
        auto maxLoad = std::get<1>(*value).load();

        to_display << "<font color=";
        if (queue_type_to_string_.at(key) == "Output Queue" || currentLoad < maxLoad * 0.8)
            to_display << "white";
        else if (currentLoad < maxLoad * 0.9)
            to_display << "orange";
        else
            to_display << "red";

        to_display << ">" << (std::get<2>(*value).load() == Device::GPU ? "GPU " : "CPU ") << queue_type_to_string_.at(key) << ":<br/>  ";
        to_display << currentLoad << "/" << maxLoad << "</font>" << "<br/>";
    }

    if (fps_map.contains(FpsType::INPUT_FPS))
    {
        to_display << fps_type_to_string_.at(FpsType::INPUT_FPS) << ":<br/>  " << input_fps_ << "<br/>";
    }

    if (fps_map.contains(FpsType::OUTPUT_FPS))
    {
        to_display << fps_type_to_string_.at(FpsType::OUTPUT_FPS) << ":<br/>  ";
        if (output_fps_ == 0)
        {
            to_display << "<font color=";
            to_display << "red";
            to_display << ">" << output_fps_ << "</font>" << "<br/>";
        }
        else
            to_display << output_fps_ << "<br/>";
    }

    if (fps_map.contains(FpsType::SAVING_FPS))
    {
        to_display << fps_type_to_string_.at(FpsType::SAVING_FPS) << ":<br/>  " << saving_fps_ << "<br/>";
    }

    if (fps_map.contains(FpsType::OUTPUT_FPS))
    {
        to_display << "Input Throughput<br/>  " << format_throughput(input_throughput_, "B/s") << "<br/>";
        to_display << "Output Throughput<br/>  " << format_throughput(output_throughput_, "Voxels/s") << "<br/>";
    }

    if (fps_map.contains(FpsType::SAVING_FPS))
    {
        to_display << "Saving Throughput<br/>  " << format_throughput(saving_throughput_, "B/s") << "<br/>";
    }

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    to_display << gpu_memory() << "<br/>";
    /* There is a memory leak on both gpu_load() and gpu_memory_controller_load(), probably linked to nvmlInit */
    to_display << gpu_load() << "<br/>";
    to_display << gpu_memory_controller_load() << "<br/>";

    // #TODO change this being called every frame to only being called to update the value if needed
    to_display << "<br/>z boundary: " << Holovibes::instance().get_boundary() << "m<br/>";

    display_info_text_function_(to_display.str());

    for (auto const& [key, value] : GSH::fast_updates_map<ProgressType>)
        update_progress_function_(key, value->first.load(), value->second.load());
}

void InformationWorker::write_information(std::ofstream& csvFile)
{
    // for fiels INPUT_QUEUE, OUTPUT_QUEUE qnd RECORD_QUEUE in GSH::fast_updates_map<QueueType> check if key present then write valuem if not write 0
    uint8_t i = 3;
    for (auto const& [key, value] : GSH::fast_updates_map<QueueType>)
    {
        if (key == QueueType::UNDEFINED)
            continue;
        
        csvFile << std::get<0>(*value).load() << ",";
        i--;
    }

    for (; i > 0; i--)
        csvFile << "0,";

    csvFile << input_fps_ << ",";
    csvFile << output_fps_ << ",";

    csvFile << input_throughput_ << ",";
    csvFile << output_throughput_ << ",";

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    csvFile << free << ",";
    csvFile << total << ",";

    csvFile << gpu_load_as_number() << ",";
    csvFile << gpu_memory_controller_load_as_number() << ",";

    // Exemple d'écriture dans le fichier CSV pour la limite z
    csvFile << Holovibes::instance().get_boundary();

    csvFile << "\n";
}

} // namespace holovibes::worker
