#include "information_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Internals

#define UPDATE_INT_OPTIONAL(map, iterator, type, target)                                                               \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = iterator->second;                                                                                     \
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

void InformationApi::get_information(Information* info)
{
    if (!info)
        throw std::runtime_error("Cannot build information: no structure provided");

    info->elapsed_time = elapsed_time_chrono_.get_milliseconds();

    auto& int_map = FastUpdatesMap::map<IntType>;
    FastUpdatesHolder<IntType>::const_iterator int_it;
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::INPUT_FPS, info->input_fps);
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::OUTPUT_FPS, info->output_fps);
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::SAVING_FPS, info->saving_fps);
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::TEMPERATURE, info->temperature);

    auto& indication_map = FastUpdatesMap::map<IndicationType>;
    FastUpdatesHolder<IndicationType>::const_iterator indication_it;
    UPDATE_STRING_OPTIONAL(indication_map, indication_it, IndicationType::IMG_SOURCE, info->img_source);
    UPDATE_STRING_OPTIONAL(indication_map, indication_it, IndicationType::INPUT_FORMAT, info->input_format);
    UPDATE_STRING_OPTIONAL(indication_map, indication_it, IndicationType::OUTPUT_FORMAT, info->output_format);

    info->progresses.clear();
    for (auto const& [key, value] : FastUpdatesMap::map<ProgressType>)
        info->progresses[key] = {value->first.load(), value->second.load()};

    info->queues.clear();
    for (auto const& [key, value] : FastUpdatesMap::map<QueueType>)
        info->queues[key] = {std::get<0>(*value).load(), std::get<1>(*value).load(), std::get<2>(*value).load()};

    this->elapsed_time_chrono_.start();
}

#pragma endregion

} // namespace holovibes::api