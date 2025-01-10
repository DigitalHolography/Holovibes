#include "information_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Internals

#define UPDATE_INT_OPTIONAL(map, iterator, type, target, value)                                                        \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = value;                                                                                                \
    else                                                                                                               \
        target.reset()

#define UPDATE_STRING_OPTIONAL(map, iterator, type, target)                                                            \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = iterator->second;                                                                                     \
    else                                                                                                               \
        target.reset()

#define UPDATE_PAIR_OPTIONAL(map, iterator, type, target)                                                              \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = {iterator->second->first, iterator->second->second};                                                  \
    else                                                                                                               \
        target.reset()

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

void InformationApi::get_slow_information(Information& info, size_t elapsed_time) { compute_fps(elapsed_time); }

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
        get_slow_information(info, elapsed_time);
        elapsed_time_chrono_.start();
    }

    auto& int_map = FastUpdatesMap::map<IntType>;
    FastUpdatesHolder<IntType>::const_iterator int_it;
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::INPUT_FPS, info.input_fps, input_fps_);
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::OUTPUT_FPS, info.output_fps, output_fps_);
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::SAVING_FPS, info.saving_fps, saving_fps_);
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::TEMPERATURE, info.temperature, temperature_);

    auto& indication_map = FastUpdatesMap::map<IndicationType>;
    FastUpdatesHolder<IndicationType>::const_iterator indication_it;
    UPDATE_STRING_OPTIONAL(indication_map, indication_it, IndicationType::IMG_SOURCE, info.img_source);
    UPDATE_STRING_OPTIONAL(indication_map, indication_it, IndicationType::INPUT_FORMAT, info.input_format);
    UPDATE_STRING_OPTIONAL(indication_map, indication_it, IndicationType::OUTPUT_FORMAT, info.output_format);

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