#include "information_api.hh"

#include <nvml.h>

#include "API.hh"

namespace holovibes::api
{

#pragma region Internals

#define UPDATE_IO_OPTIONAL(map, iterator, type, target, value1, value2)                                                \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = {value1, value2};                                                                                     \
    else                                                                                                               \
        target.reset()

#define UPDATE_STRUCT_OPTIONAL(map, iterator, type, target, ...)                                                       \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = {__VA_ARGS__};                                                                                        \
    else                                                                                                               \
        target.reset()

#define UPDATE_SIMPLE_OPTIONAL(map, iterator, type, target, value)                                                     \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = value;                                                                                                \
    else                                                                                                               \
        target.reset()

#define UPDATE_STRING_OPTIONAL(map, iterator, type, target)                                                            \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = iterator->second;                                                                                     \
    else                                                                                                               \
        target.reset()

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

std::optional<GpuInfo> get_gpu_info()
{
    nvmlUtilization_t gpu_load;

    if (get_gpu_load(&gpu_load) != NVML_SUCCESS)
        return std::nullopt;

    size_t free, total;
    cudaMemGetInfo(&free, &total);

    GpuInfo info = {gpu_load.gpu, gpu_load.memory, free, total};
    return info;
}

std::optional<RecordProgressInfo> get_record_info()
{
    std::optional<RecordProgressInfo> record_info = std::nullopt;

    auto& record_map = FastUpdatesMap::map<RecordType>;
    FastUpdatesHolder<RecordType>::const_iterator record_it;
    if ((record_it = record_map.find(RecordType::FRAME)) != record_map.end())
        record_info = {std::get<0>(*record_it->second),
                       std::get<1>(*record_it->second),
                       std::get<2>(*record_it->second)};
    return record_info;
}

void InformationApi::compute_throughput()
{
    std::shared_ptr<Queue> gpu_output_queue = API.compute.get_gpu_output_queue();
    std::shared_ptr<BatchInputQueue> input_queue = API.compute.get_input_queue();
    std::shared_ptr<Queue> frame_record_queue = Holovibes::instance().get_record_queue().load();

    unsigned int output_frame_res = 0;
    unsigned int input_frame_size = 0;
    unsigned int record_frame_size = 0;

    if (gpu_output_queue && input_queue)
    {
        output_frame_res = static_cast<unsigned int>(gpu_output_queue->get_fd().get_frame_res());
        input_frame_size = static_cast<unsigned int>(input_queue->get_fd().get_frame_size());
    }

    if (frame_record_queue)
        record_frame_size = static_cast<unsigned int>(frame_record_queue->get_fd().get_frame_size());

    input_throughput_ = input_fps_ * input_frame_size;
    output_throughput_ = output_fps_ * output_frame_res * API.transform.get_time_transformation_size();
    saving_throughput_ = saving_fps_ * record_frame_size;
}

void InformationApi::compute_fps(const long long waited_time)
{
    auto& int_map = FastUpdatesMap::map<IntType>;
    FastUpdatesHolder<IntType>::const_iterator int_it;

    if ((int_it = int_map.find(IntType::TEMPERATURE)) != int_map.end())
    {
        temperature_ = int_it->second->load();
        int_it->second->store(0);
    }

    if ((int_it = int_map.find(IntType::INPUT_FPS)) != int_map.end())
    {
        input_fps_ = static_cast<size_t>(std::round(int_it->second->load() * (1000.f / waited_time)));
        int_it->second->store(0);
    }

    if ((int_it = int_map.find(IntType::OUTPUT_FPS)) != int_map.end())
    {
        output_fps_ = static_cast<size_t>(std::round(int_it->second->load() * (1000.f / waited_time)));
        int_it->second->store(0);
    }

    if ((int_it = int_map.find(IntType::SAVING_FPS)) != int_map.end())
    {
        saving_fps_ = static_cast<size_t>(std::round(int_it->second->load() * (1000.f / waited_time)));
        int_it->second->store(0);
    }
}

#pragma endregion

#pragma region Information

void InformationApi::start_benchmark() const
{
    if (get_benchmark_mode())
        Holovibes::instance().start_benchmark();
}

void InformationApi::stop_benchmark() const { Holovibes::instance().stop_benchmark(); }

float InformationApi::get_boundary() const
{
    camera::FrameDescriptor fd = api_->input.get_input_fd();
    const float d = api_->input.get_pixel_size() * 0.000001f;
    const float n = static_cast<float>(fd.height);
    return (n * d * d) / api_->transform.get_lambda();
}

const std::string InformationApi::get_documentation_url() const
{
    return "https://ftp.espci.fr/incoming/Atlan/holovibes/manual/";
}

Information InformationApi::get_information()
{
    Information info;

    elapsed_time_chrono_.stop();
    size_t elapsed_time = elapsed_time_chrono_.get_milliseconds();
    if (elapsed_time >= 1000)
    {
        compute_fps(elapsed_time);
        compute_throughput();
        elapsed_time_chrono_.start();
    }

    auto& int_map = FastUpdatesMap::map<IntType>;
    FastUpdatesHolder<IntType>::const_iterator int_it;
    UPDATE_IO_OPTIONAL(int_map, int_it, IntType::INPUT_FPS, info.input, input_fps_, input_throughput_);
    UPDATE_IO_OPTIONAL(int_map, int_it, IntType::OUTPUT_FPS, info.output, output_fps_, output_throughput_);
    UPDATE_IO_OPTIONAL(int_map, int_it, IntType::SAVING_FPS, info.saving, saving_fps_, saving_throughput_);
    UPDATE_SIMPLE_OPTIONAL(int_map, int_it, IntType::TEMPERATURE, info.temperature, temperature_);

    auto& indication_map = FastUpdatesMap::map<IndicationType>;
    FastUpdatesHolder<IndicationType>::const_iterator indication_it;
    UPDATE_STRING_OPTIONAL(indication_map, indication_it, IndicationType::IMG_SOURCE, info.img_source);
    UPDATE_STRING_OPTIONAL(indication_map, indication_it, IndicationType::INPUT_FORMAT, info.input_format);
    UPDATE_STRING_OPTIONAL(indication_map, indication_it, IndicationType::OUTPUT_FORMAT, info.output_format);

    info.gpu_info = get_gpu_info();

    info.record_info = get_record_info();

    info.progresses.clear();
    for (auto const& [key, value] : FastUpdatesMap::map<ProgressType>)
        info.progresses[key] = {value->first.load(), value->second.load()};

    info.queues.clear();
    for (auto const& [key, value] : FastUpdatesMap::map<QueueType>)
        info.queues[key] = {std::get<0>(*value).load(), std::get<1>(*value).load(), std::get<2>(*value).load()};

    return info;
}

#pragma endregion

} // namespace holovibes::api