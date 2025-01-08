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

#define INPUT_Q_RED_COLORATION_RATIO 0.8f
#define INPUT_Q_ORANGE_COLORATION_RATIO 0.3f

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

        API.information.get_information(&information_);

        auto waited_time = chrono.get_milliseconds();
        if (waited_time >= 1000)
        {
            compute_fps(waited_time);
            std::shared_ptr<Queue> gpu_output_queue = API.compute.get_gpu_output_queue();
            std::shared_ptr<BatchInputQueue> input_queue = API.compute.get_input_queue();
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
                // Checking if there is at least 1 indication type
                if (information_.img_source || information_.input_format || information_.output_format)
                {
                    // metadata
                    benchmark_file << "Version: 0";
                    if (information_.img_source)
                        benchmark_file << ",Image Source: " << *information_.img_source.get();
                    if (information_.input_format)
                        benchmark_file << ",Input Format: " << *information_.input_format.get();
                    if (information_.output_format)
                        benchmark_file << ",Output Format: " << *information_.output_format.get();

                    for (auto const& [key, info] :
                         information_.queues) //! FIXME causes a crash on start when camera pre-selected
                        benchmark_file << "," << (info.device == Device::GPU ? "GPU " : "CPU ")
                                       << queue_type_to_string_.at(key) << " size: " << info.max_size;
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
    if (information_.temperature)
        temperature_ = information_.temperature->load();

    if (information_.input_fps)
    {
        input_fps_ = static_cast<size_t>(std::round(information_.input_fps->load() * (1000.f / waited_time)));
        information_.input_fps.get()->store(0);
    }

    if (information_.output_fps)
    {
        output_fps_ = static_cast<size_t>(std::round(information_.output_fps->load() * (1000.f / waited_time)));
        information_.output_fps->store(0); // TODO Remove
    }

    if (information_.saving_fps)
    {
        saving_fps_ = static_cast<size_t>(std::round(information_.saving_fps->load() * (1000.f / waited_time)));
        information_.saving_fps->store(0); // TODO Remove
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
    float throughput_ = throughput / (throughput > 1e9f ? 1e9f : 1e6f);
    std::string unit_ = (throughput > 1e9f ? " G" : " M") + unit;
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

const std::string get_load_color(float load,
                                 float max_load,
                                 float orange_ratio = ORANGE_COLORATION_RATIO,
                                 float red_ratio = RED_COLORATION_RATIO)
{
    const float ratio = (load / max_load);
    if (ratio < orange_ratio)
        return "white";
    if (ratio < red_ratio)
        return "orange";
    return "red";
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
    float load = static_cast<float>(gpuLoad.gpu);
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
    float load = static_cast<float>(gpuLoad.memory);
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

    float free_f = static_cast<float>(free);
    float total_f = static_cast<float>(total);

    ss << "<td style=\"color:" << get_load_color(total_f - free_f, total_f) << ";\">" << engineering_notation(free_f, 3)
       << "B free/" << engineering_notation(total_f, 3) << "B</td>";

    return ss.str();
}

void InformationWorker::display_gui_information()
{
    std::string str;
    str.reserve(512);
    std::stringstream to_display(str);

    to_display << "<table>";

    if (information_.img_source)
    {
        to_display << "<tr><td>Image Source</td><td>" << *information_.img_source.get() << "</td></tr>";
        if (information_.temperature && temperature_ != 0)
            to_display << "<tr><td>Camera Temperature</td><td>" << temperature_ << "°C</td></tr>";
    }
    if (information_.input_format)
        to_display << "<tr><td>Input Format</td><td>" << *information_.input_format.get() << "</td></tr>";
    if (information_.output_format)
        to_display << "<tr><td>Output Format</td><td>" << *information_.output_format.get() << "</td></tr>";

    if (!API.compute.get_is_computation_stopped())
    {
        for (auto const& [key, info] : information_.queues)
        {
            if (key == QueueType::UNDEFINED)
                continue;
            float currentLoad = static_cast<float>(info.current_size);
            float maxLoad = static_cast<float>(info.max_size);

            to_display << "<tr style=\"color:";
            if (key == QueueType::OUTPUT_QUEUE)
                to_display << "white";
            else if (key == QueueType::INPUT_QUEUE)
                to_display << get_load_color(currentLoad,
                                             maxLoad,
                                             INPUT_Q_ORANGE_COLORATION_RATIO,
                                             INPUT_Q_RED_COLORATION_RATIO);
            else
                to_display << get_load_color(currentLoad, maxLoad);

            to_display << ";\">";

            to_display << "<td>" << (info.device == Device::GPU ? "GPU " : "CPU ") << queue_type_to_string_.at(key)
                       << "</td>";
            to_display << "<td>" << currentLoad << "/" << maxLoad << "</td></tr>";
        }
    }

    if (information_.input_fps)
        to_display << "<tr><td>Input FPS</td><td>" << input_fps_ << "</td></tr>";

    if (information_.output_fps)
    {
        to_display << "<tr><td>Output FPS</td>";
        if (output_fps_ == 0)
            to_display << "<td style=\"color: red;\">" << output_fps_ << "</td></tr>";
        else
            to_display << "<td>" << output_fps_ << "</td></tr>";
    }

    if (information_.saving_fps)
        to_display << "<tr><td>Saving FPS</td><td>" << saving_fps_ << "</td></tr>";

    if (information_.output_fps)
    {
        to_display << "<tr><td>Input Throughput</td><td>" << format_throughput(input_throughput_, "B/s")
                   << "</td></tr>";
        to_display << "<tr><td>Output Throughput</td><td>" << format_throughput(output_throughput_, "Voxels/s")
                   << "</td></tr>";
    }

    if (information_.saving_fps)
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

    for (auto const& [key, info] : information_.progresses)
        update_progress_function_(key, info.current_size, info.max_size);
}

void InformationWorker::write_information(std::ofstream& csvFile)
{
    uint8_t i = 3;
    for (auto const& [key, info] : information_.queues)
    {
        if (key == QueueType::UNDEFINED)
            continue;

        csvFile << info.current_size << ",";
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
    csvFile << API.information.get_boundary();
    csvFile << "\n";
}

} // namespace holovibes::worker
