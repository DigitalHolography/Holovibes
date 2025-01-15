#include "record_api.hh"

#include <tuple>

#include "API.hh"

namespace holovibes::api
{

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
        {RecordMode::MOMENTS, {OutputFormat::HOLO}},
        {RecordMode::CUTS_XZ, {OutputFormat::MP4, OutputFormat::AVI}},
        {RecordMode::CUTS_YZ, {OutputFormat::MP4, OutputFormat::AVI}},
        {RecordMode::NONE, {}}}; // Just here JUST IN CASE, to avoid any potential issues

    return extension_index_map.at(mode);
}

#pragma endregion

#pragma region Eye

void RecordApi::set_recorded_eye(RecordedEyeType value) const
{
    if (!is_recording())
        UPDATE_SETTING(RecordedEye, value);
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
    nb_frames_to_record = static_cast<uint>(get_record_frame_count().value_or(0));

    if (nb_frames_to_record.load() > get_nb_frame_skip())
        nb_frames_to_record -= get_nb_frame_skip();

    // Calculate the right number of frames to record
    float pas = get_nb_frame_skip() + 1.0f;
    nb_frames_to_record = static_cast<uint>(std::floorf(static_cast<float>(nb_frames_to_record) / pas));
    if (record_mode == RecordMode::MOMENTS)
        nb_frames_to_record = nb_frames_to_record * 3;

    // Start record worker
    if (record_mode == RecordMode::CHART)
        Holovibes::instance().start_chart_record(callback);
    else
    {
        Holovibes::instance().start_frame_record(callback);

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