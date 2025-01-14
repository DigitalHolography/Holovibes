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

const std::unordered_map<QueueType, std::string> BenchmarkWorker::queue_type_to_string_ = {
    {QueueType::INPUT_QUEUE, "Input Queue"},
    {QueueType::OUTPUT_QUEUE, "Output Queue"},
    {QueueType::RECORD_QUEUE, "Record Queue"},
};

void BenchmarkWorker::run()
{
    auto benchmark_mode = GET_SETTING(BenchmarkMode);
    if (!benchmark_mode)
    {
        LOG_WARN("Benchmark mode not active, shutting down Benchmark Worker...");
        return;
    }

    std::ofstream benchmark_file;
    bool info_found = false;

    LOG_INFO("Benchmark mode active");
    std::string benchmark_file_path = settings::benchmark_dirpath + "/benchmark_NOW.csv";
    benchmark_file.open(benchmark_file_path);
    if (!benchmark_file.is_open())
        LOG_ERROR("Could not open benchmark file at " + benchmark_file_path + ", you may need to create the folder");

    while (!stop_requested_)
    {
        information_ = API.information.get_information();
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
                benchmark_file << "Input Queue,Output Queue,Record Queue,Input FPS,Output FPS,Input Throughput,Output "
                                  "Throughput,GPU memory free,GPU memory total,GPU load,GPU memory load, z_boundary\n";
                info_found = true;
            }
        }
        else
            BenchmarkWorker::write_information(benchmark_file);

        std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }

    benchmark_file.close();
    // rename file
    benchmark_file_path = settings::benchmark_dirpath + "/benchmark_" + Chrono::get_current_date_time() + ".csv";
    std::rename((settings::benchmark_dirpath + "/benchmark_NOW.csv").c_str(), benchmark_file_path.c_str());
}

void BenchmarkWorker::write_information(std::ofstream& csvFile)
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

    csvFile << (information_.input ? information_.input->fps : 0) << ",";
    csvFile << (information_.output ? information_.output->fps : 0) << ",";

    csvFile << (information_.input ? information_.input->throughput : 0) << ",";
    csvFile << (information_.input ? information_.output->throughput : 0) << ",";

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    csvFile << free << ",";
    csvFile << total << ",";

    if (information_.gpu_info)
    {
        csvFile << information_.gpu_info->gpu << ",";
        csvFile << information_.gpu_info->memory << ",";
    }

    // Exemple d'écriture dans le fichier CSV pour la limite z
    csvFile << API.information.get_boundary();
    csvFile << "\n";
}

} // namespace holovibes::worker
