#include "record_api.hh"

#include <tuple>

#include "API.hh"
#include "notifier.hh"

namespace holovibes::api
{

#pragma region Record Mode

void RecordApi::set_record_mode_enum(RecordMode value) const
{
    stop_record();

    set_record_mode(value);

    // Attempt to initialize compute pipe for non-CHART record modes
    if (get_record_mode() != RecordMode::CHART)
    {
        try
        {
            auto pipe = api_->compute.get_compute_pipe();
            if (is_recording())
                stop_record();

            Holovibes::instance().init_record_queue();
            LOG_DEBUG("Pipe initialized");
        }
        catch (const std::exception& e)
        {
            (void)e; // Suppress warning in case debug log is disabled
            LOG_DEBUG("Pipe not initialized: {}", e.what());
        }
    }
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

    return true;
}

void RecordApi::start_record(std::function<void()> callback) const
{
    if (!start_record_preconditions()) // Check if the record can be started
        return;

    RecordMode record_mode = GET_SETTING(RecordMode);

    if (record_mode == RecordMode::CHART)
        Holovibes::instance().start_chart_record(callback);
    else
        Holovibes::instance().start_frame_record(callback);

    // Notify the changes
    NotifierManager::notify<bool>("acquisition_started", true); // notifying MainWindow
}

void RecordApi::stop_record() const
{
    LOG_FUNC();

    if (api_->compute.get_is_computation_stopped())
        return;

    auto record_mode = GET_SETTING(RecordMode);

    if (record_mode == RecordMode::CHART)
        Holovibes::instance().stop_chart_record();
    else if (record_mode != RecordMode::NONE)
    {
        api_->compute.get_compute_pipe()->request(ICS::DisableFrameRecord);
        Holovibes::instance().stop_frame_record();
    }
}

bool RecordApi::is_recording() const { return Holovibes::instance().is_recording(); }

#pragma endregion

#pragma region Buffer

void RecordApi::set_record_queue_location(Device device) const
{
    // we check since this function is always triggered when we save the advanced settings, even if the location was not
    // modified
    if (get_record_queue_location() != device)
    {
        UPDATE_SETTING(RecordQueueLocation, device);

        if (is_recording())
            stop_record();

        Holovibes::instance().init_record_queue();
    }
}

void RecordApi::set_record_buffer_size(uint value) const
{
    // since this function is always triggered when we save the advanced settings, even if the location was not modified
    if (get_record_buffer_size() != value)
    {
        UPDATE_SETTING(RecordBufferSize, value);

        if (is_recording())
            stop_record();

        if (!api_->compute.get_is_computation_stopped())
            Holovibes::instance().init_record_queue();
    }
}

#pragma endregion

} // namespace holovibes::api