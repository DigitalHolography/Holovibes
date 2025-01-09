#include <fstream>
#include "api.hh"
#include "holovibes.hh"
#include "icompute.hh"
#include "gpu_stats.hh"
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
