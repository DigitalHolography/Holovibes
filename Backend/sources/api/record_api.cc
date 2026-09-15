#include "record_api.hh"

#include <tuple>
#include <limits>

#include "API.hh"

namespace holovibes::api
{
namespace
{
size_t target_frame_count(std::optional<size_t> count, size_t frame_skip, RecordMode mode)
{
    if (!count)
        return 0;
    const size_t after_offset = *count > frame_skip ? *count - frame_skip : *count;
    const size_t frames = after_offset / (frame_skip + 1);
    return mode == RecordMode::MOMENTS ? frames * 3 : frames;
}
} // namespace

#pragma region Record Mode

ApiCode RecordApi::set_record_mode(RecordMode value) const
{
    if (value == get_record_mode())
        return ApiCode::NO_CHANGE;

    if (is_recording())
        stop_record();

    UPDATE_SETTING(RecordMode, value);

    if (get_record_mode() == RecordMode::CHART)
        return ApiCode::OK;

    // Update the record queue only if an input source is available since otherwise input_fd is not set
    if (api_->input.get_import_type() != ImportType::None)
        Holovibes::instance().init_record_queue();

    return ApiCode::OK;
}

std::vector<OutputFormat> RecordApi::get_supported_formats(RecordMode mode) const
{
    static const std::map<RecordMode, std::vector<OutputFormat>> extension_index_map = {
        {RecordMode::RAW, {OutputFormat::HOLO}},
        {RecordMode::CHART, {OutputFormat::CSV, OutputFormat::TXT}},
        {RecordMode::HOLOGRAM, {OutputFormat::HOLO, OutputFormat::MP4, OutputFormat::AVI}},
        {RecordMode::MOMENTS, {OutputFormat::H5}},
        {RecordMode::CUTS_XZ, {OutputFormat::MP4, OutputFormat::AVI}},
        {RecordMode::CUTS_YZ, {OutputFormat::MP4, OutputFormat::AVI}},
        {RecordMode::OCT_CUBE, {OutputFormat::H5}},
        {RecordMode::OCT_CUBE_FLOAT, {OutputFormat::H5}},
        {RecordMode::NONE, {}}}; // Just here JUST IN CASE, to avoid any potential issues

    return extension_index_map.at(mode);
}

#pragma endregion

#pragma region Eye

ApiCode RecordApi::set_recorded_eye(RecordedEyeType value) const
{
    if (API.input.get_import_type() != ImportType::Camera || value == GET_SETTING(RecordedEye))
        return ApiCode::NO_CHANGE;

    UPDATE_SETTING(RecordedEye, value);

    return ApiCode::OK;
}

#pragma endregion

#pragma region Recording

RecordProgress RecordApi::get_record_progress() const
{
    auto fast_update_progress_entry = FastUpdatesMap::map<RecordType>.get_or_create_entry(RecordType::FRAME);
    std::atomic<uint>& nb_frame_acquired = std::get<0>(*fast_update_progress_entry);
    std::atomic<uint>& nb_frames_recorded = std::get<1>(*fast_update_progress_entry);
    std::atomic<uint>& nb_frames_to_record = std::get<2>(*fast_update_progress_entry);

    return {nb_frame_acquired.load(), nb_frames_recorded.load(), nb_frames_to_record.load()};
}

bool RecordApi::start_record_preconditions() const
{
    if (get_record_mode() == RecordMode::CHART && get_record_frame_count() == std::nullopt)
    {
        LOG_ERROR("Number of frames must be activated");
        return false;
    }

    if (api_->transform.get_batch_size() > get_record_buffer_size())
    {
        LOG_ERROR("Batch size must be lower than record queue size");
        return false;
    }

    if (get_async_record_ram_gib() != 0 && get_record_mode() != RecordMode::CHART)
    {
        auto queue = Holovibes::instance().get_record_queue().load();
        if (!queue && api_->input.get_import_type() != ImportType::None)
        {
            Holovibes::instance().init_record_queue();
            queue = Holovibes::instance().get_record_queue().load();
        }
        if (!queue)
        {
            LOG_ERROR("Record queue is unavailable");
            return false;
        }
        const size_t frame_size = queue->get_fd().get_frame_size();
        const size_t multiplier = get_record_mode() == RecordMode::MOMENTS
                                      ? 3
                                      : (get_record_mode() == RecordMode::OCT_CUBE ||
                                         get_record_mode() == RecordMode::OCT_CUBE_FLOAT)
                                            ? api_->transform.get_time_transformation_size()
                                            : 1;
        if (frame_size == 0 || multiplier == 0 || multiplier > std::numeric_limits<size_t>::max() / frame_size)
            return false;

        const size_t frames = target_frame_count(get_record_frame_count(), get_nb_frame_skip(), get_record_mode());
        const bool fits = get_record_frame_count()
                              ? (frames <= std::numeric_limits<size_t>::max() / frame_size &&
                                 Holovibes::instance().get_async_record_writer().can_reserve(frames * frame_size))
                              : Holovibes::instance().get_async_record_writer().can_reserve_unbounded(frame_size * multiplier);
        if (!fits)
        {
            LOG_ERROR("Not enough queued-save RAM is available for this recording; wait for saves or increase the RAM budget");
            return false;
        }
    }

    return true;
}

ApiCode RecordApi::start_record(std::function<void()> callback) const
{
    if (!start_record_preconditions()) // Check if the record can be started
        return ApiCode::FAILURE;

    RecordMode record_mode = GET_SETTING(RecordMode);

    // Reset recording counter
    auto fast_update_progress_entry = FastUpdatesMap::map<RecordType>.get_or_create_entry(RecordType::FRAME);
    std::atomic<uint>& nb_frames_to_record = std::get<2>(*fast_update_progress_entry);

    std::get<0>(*fast_update_progress_entry) = 0; // Frames acquired
    std::get<1>(*fast_update_progress_entry) = 0; // Frames recorded
    nb_frames_to_record = static_cast<uint>(target_frame_count(get_record_frame_count(), get_nb_frame_skip(), record_mode));

    // Start record worker
    if (record_mode == RecordMode::CHART)
        Holovibes::instance().start_chart_record(callback);
    else
    {
        if (!Holovibes::instance().start_frame_record(callback))
        {
            LOG_ERROR("Queued-save RAM reservation failed; wait for pending saves");
            return ApiCode::FAILURE;
        }

        set_frame_acquisition_enabled(true);
    }

    return ApiCode::OK;
}

ApiCode RecordApi::stop_record() const
{
    LOG_FUNC();

    if (api_->compute.get_is_computation_stopped())
        return ApiCode::NOT_STARTED;

    auto record_mode = GET_SETTING(RecordMode);

    if (record_mode == RecordMode::CHART)
        Holovibes::instance().stop_chart_record();
    else if (record_mode != RecordMode::NONE)
    {
        api_->compute.get_compute_pipe()->request(ICS::DisableFrameRecord);
        Holovibes::instance().stop_frame_record();
    }

    return ApiCode::OK;
}

bool RecordApi::is_recording() const { return Holovibes::instance().is_recording(); }

#pragma endregion

#pragma region Buffer

ApiCode RecordApi::set_record_queue_location(Device device) const
{
    if (get_record_queue_location() == device)
        return ApiCode::NO_CHANGE;

    if (is_recording())
        stop_record();

    UPDATE_SETTING(RecordQueueLocation, device);

    if (api_->input.get_import_type() != ImportType::None)
        Holovibes::instance().init_record_queue();

    return ApiCode::OK;
}

ApiCode RecordApi::set_record_buffer_size(uint value) const
{
    if (get_record_buffer_size() == value)
        return ApiCode::NO_CHANGE;

    UPDATE_SETTING(RecordBufferSize, value);

    if (is_recording())
        stop_record();

    if (api_->input.get_import_type() != ImportType::None)
        Holovibes::instance().init_record_queue();

    return ApiCode::OK;
}

#pragma endregion

} // namespace holovibes::api
