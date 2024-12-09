#include "record_api.hh"

#include "API.hh"

namespace holovibes::api
{

#pragma region Record Mode

void RecordApi::set_record_mode_enum(RecordMode value)
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

            init_record_queue();
            LOG_DEBUG("Pipe initialized");
        }
        catch (const std::exception& e)
        {
            (void)e; // Suppress warning in case debug log is disabled
            LOG_DEBUG("Pipe not initialized: {}", e.what());
        }
    }
}

std::vector<OutputFormat> RecordApi::get_supported_formats(RecordMode mode)
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

bool RecordApi::start_record_preconditions()
{
    std::optional<size_t> nb_frames_to_record = get_record_frame_count();
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

void RecordApi::start_record(std::function<void()> callback)
{
    if (!start_record_preconditions()) // Check if the record can be started
        return;

    RecordMode record_mode = GET_SETTING(RecordMode);

    if (record_mode == RecordMode::CHART)
        start_chart_record(callback);
    else
        start_frame_record(callback);

    // Notify the changes
    NotifierManager::notify<RecordMode>("record_start", record_mode); // notifying lightUI
    NotifierManager::notify<bool>("acquisition_started", true);       // notifying MainWindow
}

void RecordApi::stop_record()
{
    LOG_FUNC();

    auto record_mode = GET_SETTING(RecordMode);

    if (record_mode == RecordMode::CHART)
        stop_chart_record();
    else if (record_mode != RecordMode::NONE)
        stop_frame_record();

    // Notify the changes
    NotifierManager::notify<RecordMode>("record_stop", record_mode);
}

#pragma endregion

#pragma region Buffer

void RecordApi::set_record_queue_location(Device device)
{
    // we check since this function is always triggered when we save the advanced settings, even if the location was not
    // modified
    if (get_record_queue_location() != device)
    {
        UPDATE_SETTING(RecordQueueLocation, device);

        if (is_recording())
            stop_record();

        init_record_queue();
    }
}

void RecordApi::set_record_buffer_size(uint value)
{
    // since this function is always triggered when we save the advanced settings, even if the location was not modified
    if (get_record_buffer_size() != value)
    {
        UPDATE_SETTING(RecordBufferSize, value);

        if (is_recording())
            stop_record();

        init_record_queue();
    }
}

#pragma endregion

bool RecordApi::is_recording() const { return frame_record_worker_controller_.is_running(); }

void RecordApi::init_record_queue()
{
    auto device = get_record_queue_location();
    auto record_mode = get_record_mode();
    auto input_queue = Holovibes::instance().get_input_queue();
    auto gpu_output_queue = Holovibes::instance().get_gpu_output_queue();

    switch (record_mode)
    {
    case RecordMode::RAW:
    {
        if (!input_queue)
        {
            LOG_DEBUG("Cannot create record queue : input queue not created");
            return;
        }
        LOG_DEBUG("RecordMode = Raw");
        if (!record_queue_.load())
            record_queue_ = std::make_shared<Queue>(input_queue->get_fd(),
                                                    get_record_buffer_size(),
                                                    QueueType::RECORD_QUEUE,
                                                    device);
        else
            record_queue_.load()->rebuild(input_queue->get_fd(),
                                          get_record_buffer_size(),
                                          Holovibes::instance().get_cuda_streams().recorder_stream,
                                          device);

        LOG_DEBUG("Record queue allocated");
        break;
    }
    case RecordMode::HOLOGRAM:
    {
        LOG_DEBUG("RecordMode = Hologram");

        if (!gpu_output_queue)
        {
            api_->compute.pipe_refresh();
            return;
        }

        auto record_fd = gpu_output_queue->get_fd();
        if (record_fd.depth == camera::PixelDepth::Bits8)
            record_fd.depth = camera::PixelDepth::Bits16;
        if (!record_queue_.load())
        {
            record_queue_ =
                std::make_shared<Queue>(record_fd, get_record_buffer_size(), QueueType::RECORD_QUEUE, device);
        }
        else
            record_queue_.load()->rebuild(record_fd,
                                          get_record_buffer_size(),
                                          Holovibes::instance().get_cuda_streams().recorder_stream,
                                          device);
        LOG_DEBUG("Record queue allocated");
        break;
    }
    case RecordMode::CUTS_YZ:
    case RecordMode::CUTS_XZ:
    {
        LOG_DEBUG("RecordMode = CUTS");
        camera::FrameDescriptor fd_xyz = gpu_output_queue->get_fd();
        fd_xyz.depth = camera::PixelDepth::Bits16; // Size of ushort
        if (record_mode == RecordMode::CUTS_XZ)
            fd_xyz.height = api_->transform.get_time_transformation_size();
        else
            fd_xyz.width = api_->transform.get_time_transformation_size();

        if (!record_queue_.load())
            record_queue_ = std::make_shared<Queue>(fd_xyz, get_record_buffer_size(), QueueType::RECORD_QUEUE, device);
        else
            record_queue_.load()->rebuild(fd_xyz,
                                          get_record_buffer_size(),
                                          Holovibes::instance().get_cuda_streams().recorder_stream,
                                          device);
        LOG_DEBUG("Record queue allocated");
        break;
    }
    case RecordMode::MOMENTS:
    {
        LOG_DEBUG("RecordMode = Moments");
        camera::FrameDescriptor record_fd = input_queue->get_fd();
        record_fd.depth = camera::PixelDepth::Bits32;

        if (!record_queue_.load())
            record_queue_ =
                std::make_shared<Queue>(record_fd, get_record_buffer_size(), QueueType::RECORD_QUEUE, device);
        else
            record_queue_.load()->rebuild(record_fd,
                                          get_record_buffer_size(),
                                          Holovibes::instance().get_cuda_streams().recorder_stream,
                                          device);
        LOG_DEBUG("Record queue allocated");
        break;
    }
    default:
    {
        LOG_DEBUG("RecordMode = None");
        break;
    }
    }
}

} // namespace holovibes::api