#include <fstream>
#include "api.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "tools.hh"
#include "chrono.hh"
#include <cuda_runtime.h>
#include <chrono>
#include "fast_updates_holder.hh"
#include <nvml.h>
#include "logger.hh"
#include "spdlog/spdlog.h"

namespace holovibes::worker
{
using MutexGuard = std::lock_guard<std::mutex>;

#define RED_COLORATION_RATIO 0.9f
#define ORANGE_COLORATION_RATIO 0.7f

const std::unordered_map<IndicationType, std::string> InformationWorker::indication_type_to_string_ = {
    {IndicationType::IMG_SOURCE, "Image Source"},
    {IndicationType::INPUT_FORMAT, "Input Format"},
    {IndicationType::OUTPUT_FORMAT, "Output Format"},
};

const std::unordered_map<IntType, std::string> InformationWorker::fps_type_to_string_ = {
    {IntType::INPUT_FPS, "Input FPS"},
    {IntType::OUTPUT_FPS, "Output FPS"},
    {IntType::SAVING_FPS, "Saving FPS"},
    {IntType::TEMPERATURE, "Camera Temperature"},
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

    auto benchmark_mode = GET_SETTING(BenchmarkMode);
    std::ofstream benchmark_file;
    bool info_found = false;

    if (benchmark_mode)
    {
        LOG_INFO("Benchmark mode active");
        std::string benchmark_file_path = settings::benchmark_dirpath + "/benchmark_NOW.csv";
        benchmark_file.open(benchmark_file_path);
        if (!benchmark_file.is_open())
            LOG_ERROR("Could not open benchmark file at " + benchmark_file_path +
                      ", you may need to create the folder");
    }

    while (!stop_requested_)
    {
        chrono.stop();

        auto waited_time = chrono.get_milliseconds();
        if (waited_time >= 1000)
        {
            compute_fps(waited_time);
            std::shared_ptr<Queue> gpu_output_queue = api::get_gpu_output_queue();
            std::shared_ptr<BatchInputQueue> input_queue = api::get_input_queue();
            std::shared_ptr<Queue> frame_record_queue = Holovibes::instance().get_record_queue().load();

            if (gpu_output_queue && input_queue)
            {
                output_frame_res = static_cast<unsigned int>(gpu_output_queue->get_fd().get_frame_res());
                input_frame_size = static_cast<unsigned int>(input_queue->get_fd().get_frame_size());
            }

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
                if (!FastUpdatesMap::map<IndicationType>.empty())
                {
                    // metadata
                    benchmark_file << "Version: 0";
                    for (auto const& [key, value] : FastUpdatesMap::map<IndicationType>)
                        benchmark_file << "," << indication_type_to_string_.at(key) << ": " << *value;
                    for (auto const& [key, value] :
                         FastUpdatesMap::map<QueueType>) //! FIXME causes a crash on start when camera pre-selected
                        benchmark_file << "," << (std::get<2>(*value).load() == Device::GPU ? "GPU " : "CPU ")
                                       << queue_type_to_string_.at(key) << " size: " << std::get<1>(*value).load();
                    benchmark_file << "\n";
                    // 11 headers
                    benchmark_file
                        << "Input Queue,Output Queue,Record Queue,Input FPS,Output FPS,Input Throughput,Output "
                           "Throughput,GPU memory free,GPU memory total,GPU load,GPU memory load, z_boundary\n";
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
        std::string benchmark_file_path =
            settings::benchmark_dirpath + "/benchmark_" + Chrono::get_current_date_time() + ".csv";
        std::rename((settings::benchmark_dirpath + "/benchmark_NOW.csv").c_str(), benchmark_file_path.c_str());
    }
}

void InformationWorker::compute_fps(const long long waited_time)
{
    auto& fps_map = FastUpdatesMap::map<IntType>;
    FastUpdatesHolder<IntType>::const_iterator it;
    if ((it = fps_map.find(IntType::TEMPERATURE)) != fps_map.end())
        temperature_ = it->second->load();

    if ((it = fps_map.find(IntType::INPUT_FPS)) != fps_map.end())
        input_fps_ = it->second->load();

    if ((it = fps_map.find(IntType::OUTPUT_FPS)) != fps_map.end())
    {
        output_fps_ = std::round(it->second->load() * (1000.f / waited_time));
        it->second->store(0); // TODO Remove
    }

    if ((it = fps_map.find(IntType::SAVING_FPS)) != fps_map.end())
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

int get_gpu_load(nvmlUtilization_t* gpuLoad)
{
    nvmlDevice_t device;

    // Initialize NVML
    if (nvmlInit() != NVML_SUCCESS)
        return -1;

    // Get the device handle (assuming only one GPU is present)
    if (nvmlDeviceGetHandleByIndex(0, &device) != NVML_SUCCESS)
    {
        nvmlShutdown();
        return -1;
    }

    // Query GPU load
    if (nvmlDeviceGetUtilizationRates(device, gpuLoad) != NVML_SUCCESS)
    {
        nvmlShutdown();
        return -1;
    }

    // Shutdown NVML
    return nvmlShutdown();
}

const std::string get_load_color_custom(float load, float max_load, float orange_ratio, float red_ratio)
{
    const float ratio = (load / max_load);
    if (ratio < orange_ratio)
        return "white";
    if (ratio < red_ratio)
        return "orange";
    return "red";
}

const std::string get_load_color(float load, float max_load)
{
    return get_load_color_custom(load, max_load, ORANGE_COLORATION_RATIO, RED_COLORATION_RATIO);
}

const std::string get_percentage_color_custom(float percentage, float orange_ratio, float red_ratio)
{
    return get_load_color_custom(percentage, 100, orange_ratio, red_ratio);
}

const std::string get_percentage_color(float percentage) { return get_load_color(percentage, 100); }

std::string gpu_load()
{
    nvmlUtilization_t gpuLoad;
    std::stringstream ss;
    ss << "<td>GPU load</td>";

    if (get_gpu_load(&gpuLoad) != NVML_SUCCESS)
    {
        ss << "<td>Could not load GPU usage</td>";
        return ss.str();
    }

    // Print GPU load
    auto load = gpuLoad.gpu;
    ss << "<td style=\"color:" << get_percentage_color(load) << ";\">" << load << "%</td>";

    return ss.str();
}

std::string gpu_load_as_number()
{
    nvmlUtilization_t gpuLoad;

    if (get_gpu_load(&gpuLoad) != NVML_SUCCESS)
        return "Could not load GPU usage";

    return std::to_string(gpuLoad.gpu);
}

std::string gpu_memory_controller_load()
{
    nvmlUtilization_t gpuLoad;
    std::stringstream ss;
    ss << "<td style=\"padding-right: 15px\">VRAM controller load</td>";

    if (get_gpu_load(&gpuLoad) != NVML_SUCCESS)
    {
        ss << "<td>Could not load GPU usage</td>";
        return ss.str();
    }

    // Print GPU memory load
    auto load = gpuLoad.memory;
    ss << "<td style=\"color:" << get_percentage_color(load) << ";\">" << load << "%</td>";

    return ss.str();
}

std::string gpu_memory_controller_load_as_number()
{
    nvmlUtilization_t gpuLoad;

    if (get_gpu_load(&gpuLoad) != NVML_SUCCESS)
        return "Could not load GPU usage";

    return std::to_string(gpuLoad.memory);
}

std::string gpu_memory()
{
    std::stringstream ss;
    ss << "<td>VRAM</td>";
    size_t free, total;
    cudaMemGetInfo(&free, &total);

    ss << "<td style=\"color:" << get_load_color(total - free, total) << ";\">" << engineering_notation(free, 3)
       << "B free/" << engineering_notation(total, 3) << "B</td>";

    return ss.str();
}

void InformationWorker::display_gui_information()
{
    std::string str;
    str.reserve(512);
    std::stringstream to_display(str);
    auto& fps_map = FastUpdatesMap::map<IntType>;

    to_display << "<table>";

    for (auto const& [key, value] : FastUpdatesMap::map<IndicationType>)
    {
        to_display << "<tr><td>" << indication_type_to_string_.at(key) << "</td><td>" << *value << "</td></tr>";
        if (key == IndicationType::IMG_SOURCE && fps_map.contains(IntType::TEMPERATURE) && temperature_ != 0)
        {
            to_display << "<tr><td>" << fps_type_to_string_.at(IntType::TEMPERATURE) << "</td><td>" << temperature_
                       << "°C</td></tr>";
        }
    }

    for (auto const& [key, value] : FastUpdatesMap::map<QueueType>)
    {
        if (key == QueueType::UNDEFINED)
            continue;
        auto currentLoad = std::get<0>(*value).load();
        auto maxLoad = std::get<1>(*value).load();

        to_display << "<tr style=\"color:";
        if (queue_type_to_string_.at(key) == "Output Queue")
            to_display << "white";
        else if (key == QueueType::INPUT_QUEUE)
            to_display << get_load_color_custom(currentLoad, maxLoad, 0.3, 0.8);
        else
            to_display << get_load_color(currentLoad, maxLoad);

        to_display << ";\">";

        to_display << "<td>" << (std::get<2>(*value).load() == Device::GPU ? "GPU " : "CPU ")
                   << queue_type_to_string_.at(key) << "</td>";
        to_display << "<td>" << currentLoad << "/" << maxLoad << "</td></tr>";
    }

    if (fps_map.contains(IntType::INPUT_FPS))
        to_display << "<tr><td>" << fps_type_to_string_.at(IntType::INPUT_FPS) << "</td><td>" << input_fps_
                   << "</td></tr>";

    if (fps_map.contains(IntType::OUTPUT_FPS))
    {
        to_display << "<tr><td>" << fps_type_to_string_.at(IntType::OUTPUT_FPS) << "</td>";
        if (output_fps_ == 0)
            to_display << "<td style=\"color: red;\">" << output_fps_ << "</td></tr>";
        else
            to_display << "<td>" << output_fps_ << "</td></tr>";
    }

    if (fps_map.contains(IntType::SAVING_FPS))
        to_display << "<tr><td>" << fps_type_to_string_.at(IntType::SAVING_FPS) << "</td><td>" << saving_fps_
                   << "</td></tr>";

    if (fps_map.contains(IntType::OUTPUT_FPS))
    {
        to_display << "<tr><td>Input Throughput</td><td>" << format_throughput(input_throughput_, "B/s")
                   << "</td></tr>";
        to_display << "<tr><td>Output Throughput</td><td>" << format_throughput(output_throughput_, "Voxels/s")
                   << "</td></tr>";
    }

    if (fps_map.contains(IntType::SAVING_FPS))
    {
        to_display << "<tr><td>Saving Throughput</td><td>  " << format_throughput(saving_throughput_, "B/s")
                   << "</td></tr>";
    }

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    to_display << "<tr>" << gpu_memory() << "</tr>";
    /* There is a memory leak on both gpu_load() and gpu_memory_controller_load(), probably linked to nvmlInit */
    to_display << "<tr>" << gpu_load() << "</tr>";
    to_display << "<tr>" << gpu_memory_controller_load() << "</tr>";

    to_display << "</table>";

    display_info_text_function_(to_display.str());

    for (auto const& [key, value] : FastUpdatesMap::map<ProgressType>)
        update_progress_function_(key, value->first.load(), value->second.load());
}

void InformationWorker::write_information(std::ofstream& csvFile)
{
    // for fiels INPUT_QUEUE, OUTPUT_QUEUE qnd RECORD_QUEUE in FastUpdatesMap::map<QueueType> check if key present
    // then write valuem if not write 0
    uint8_t i = 3;
    for (auto const& [key, value] : FastUpdatesMap::map<QueueType>)
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
