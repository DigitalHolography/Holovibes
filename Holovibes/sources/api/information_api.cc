#include "information_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Internals

#define UPDATE_INT_OPTIONAL(map, iterator, type, target)                                                               \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = iterator->second->load();                                                                             \
    else                                                                                                               \
        target.reset()

#define UPDATE_PAIR_OPTIONAL(map, iterator, type, target)                                                              \
    if ((iterator = map.find(type)) != map.end())                                                                      \
        target = {type, iterator->second->first, iterator->second->second};                                            \
    else                                                                                                               \
        target.reset()

#pragma endregion

#pragma region Information

void InformationApi::start_information_display() const { Holovibes::instance().start_information_display(); }

float InformationApi::get_boundary() const { return Holovibes::instance().get_boundary(); }

const std::string InformationApi::get_documentation_url() const
{
    return "https://ftp.espci.fr/incoming/Atlan/holovibes/manual/";
}

void get_information(Information* info)
{
    if (!info)
        throw std::runtime_error("Cannot build information: no structure provided");

    auto& int_map = FastUpdatesMap::map<IntType>;
    FastUpdatesHolder<IntType>::const_iterator int_it;
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::INPUT_FPS, info->input_fps);
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::OUTPUT_FPS, info->output_fps);
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::SAVING_FPS, info->saving_fps);
    UPDATE_INT_OPTIONAL(int_map, int_it, IntType::TEMPERATURE, info->temperature);

    auto& progress_map = FastUpdatesMap::map<ProgressType>;
    FastUpdatesHolder<ProgressType>::const_iterator progress_it;
    UPDATE_PAIR_OPTIONAL(progress_map, progress_it, ProgressType::FILE_READ, info->file_read_progress);
    // The two 'record' entries are in the same 'slot' because they will never be active at the same time.
    UPDATE_PAIR_OPTIONAL(progress_map, progress_it, ProgressType::FRAME_RECORD, info->record_progress);
    // Chart not as a macro because if FRAME_RECORD is active, using a macro would .reset() the FRAME_RECORD data.
    if ((progress_it = progress_map.find(ProgressType::CHART_RECORD)) != progress_map.end())
        info->record_progress = {ProgressType::CHART_RECORD, progress_it->second->first, progress_it->second->second};

    info->queues.clear();
    for (auto const& [key, value] : FastUpdatesMap::map<QueueType>)
        info->queues[key] = {std::get<0>(*value).load(), std::get<1>(*value).load(), std::get<2>(*value).load()};
}

#pragma endregion

} // namespace holovibes::api