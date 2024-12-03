#include "record_api.hh"

namespace holovibes::api
{

#pragma region Record Mode

void set_record_mode_enum(RecordMode value)
{
    stop_record();

    set_record_mode(value);

    // Attempt to initialize compute pipe for non-CHART record modes
    if (get_record_mode() != RecordMode::CHART)
    {
        try
        {
            auto pipe = get_compute_pipe();
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

std::vector<OutputFormat> get_supported_formats(RecordMode mode)
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

#pragma region Recording

bool start_record_preconditions()
{
    std::optional<size_t> nb_frames_to_record = api::get_record_frame_count();
    bool nb_frame_checked = nb_frames_to_record.has_value();

    if (!nb_frame_checked)
        nb_frames_to_record = std::nullopt;

    if (get_record_mode() == RecordMode::CHART && nb_frames_to_record == std::nullopt)
    {
        LOG_ERROR("Number of frames must be activated");
        return false;
    }

    return true;
}

void start_record(std::function<void()> callback)
{
    if (!start_record_preconditions()) // Check if the record can be started
        return;

    RecordMode record_mode = GET_SETTING(RecordMode);

    if (record_mode == RecordMode::CHART)
        Holovibes::instance().start_chart_record(callback);
    else
        Holovibes::instance().start_frame_record(callback);

    // Notify the changes
    NotifierManager::notify<RecordMode>("record_start", record_mode); // notifying lightUI
    NotifierManager::notify<bool>("acquisition_started", true);       // notifying MainWindow
}

void stop_record()
{
    LOG_FUNC();

    auto record_mode = GET_SETTING(RecordMode);

    if (record_mode == RecordMode::CHART)
        Holovibes::instance().stop_chart_record();
    else if (record_mode != RecordMode::NONE)
        Holovibes::instance().stop_frame_record();

    // Notify the changes
    NotifierManager::notify<RecordMode>("record_stop", record_mode);
}

bool is_recording() { return Holovibes::instance().is_recording(); }

#pragma endregion

#pragma region Buffer

void set_record_queue_location(Device device)
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

void set_record_buffer_size(uint value)
{
    // since this function is always triggered when we save the advanced settings, even if the location was not modified
    if (get_record_buffer_size() != value)
    {
        UPDATE_SETTING(RecordBufferSize, value);

        if (is_recording())
            stop_record();

        Holovibes::instance().init_record_queue();
    }
}

#pragma endregion

} // namespace holovibes::api