#include "holovibes.hh"
#include "queue.hh"

#include "camera_dll.hh"
#include "tools.hh"
#include "logger.hh"
#include "holo_file.hh"
#include "icompute.hh"
#include "API.hh"

namespace holovibes
{
using camera::FrameDescriptor;

Holovibes& Holovibes::instance()
{
    static Holovibes instance;
    return instance;
}

bool Holovibes::is_recording() const { return frame_record_worker_controller_.is_running(); }

void Holovibes::init_input_queue(const camera::FrameDescriptor& fd, const unsigned int input_queue_size)
{
    if (!input_queue_.load())
        input_queue_ = std::make_shared<BatchInputQueue>(input_queue_size, API.transform.get_batch_size(), fd);
    else
        input_queue_.load()->rebuild(fd, input_queue_size, API.transform.get_batch_size(), Device::GPU);
    LOG_DEBUG("Input queue allocated");
}

void Holovibes::init_record_queue()
{
    auto& api = API;
    auto device = api.record.get_record_queue_location();
    auto size = api.record.get_record_buffer_size();
    auto record_mode = api.record.get_record_mode();

    camera::FrameDescriptor fd = api.input.get_input_fd();

    switch (record_mode)
    {
    case RecordMode::RAW:
    {
        LOG_DEBUG("RecordMode = Raw");
        break;
    }
    case RecordMode::HOLOGRAM:
    {
        LOG_DEBUG("RecordMode = Hologram");

        fd.depth = camera::PixelDepth::Bits16;
        if (api.compute.get_img_type() == ImgType::Composite)
            fd.depth = camera::PixelDepth::Bits48;

        break;
    }
    case RecordMode::CUTS_YZ:
    case RecordMode::CUTS_XZ:
    {
        LOG_DEBUG("RecordMode = CUTS");

        fd.depth = camera::PixelDepth::Bits16; // Size of ushort
        if (record_mode == RecordMode::CUTS_XZ)
            fd.height = api.transform.get_time_transformation_size();
        else
            fd.width = api.transform.get_time_transformation_size();

        break;
    }
    case RecordMode::MOMENTS:
    {
        LOG_DEBUG("RecordMode = Moments");
        fd.depth = camera::PixelDepth::Bits32;

        break;
    }
    default:
    {
        LOG_DEBUG("RecordMode = None");
        return;
    }
    }

    if (!record_queue_.load())
        record_queue_ = std::make_shared<Queue>(fd, size, QueueType::RECORD_QUEUE, device);
    else
        record_queue_.load()->rebuild(fd, size, get_cuda_streams().recorder_stream, device);

    LOG_DEBUG("Record queue allocated");
}

void Holovibes::start_file_frame_read(const std::function<void()>& callback)
{
    CHECK(input_queue_.load() != nullptr);

    file_read_worker_controller_.set_callback(callback);
    file_read_worker_controller_.set_error_callback(error_callback_);
    file_read_worker_controller_.set_priority(THREAD_READER_PRIORITY);

    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    file_read_worker_controller_.start(input_queue_, all_settings);
}

void Holovibes::start_camera_frame_read()
{
    try
    {
        camera_read_worker_controller_.set_error_callback(error_callback_);
        camera_read_worker_controller_.set_priority(THREAD_READER_PRIORITY);
        camera_read_worker_controller_.start(active_camera_, input_queue_);
    }
    catch (std::exception& e)
    {
        LOG_ERROR("Error at camera frame read start worker. (Exception: {})", e.what());
        stop_frame_read();
        throw;
    }
}

void Holovibes::stop_frame_read()
{
    LOG_FUNC();
    camera_read_worker_controller_.stop();
    file_read_worker_controller_.stop();

    while (camera_read_worker_controller_.is_running() || file_read_worker_controller_.is_running())
        continue;

    input_queue_.store(nullptr);
}

void Holovibes::start_frame_record(const std::function<void()>& callback)
{
    if (!record_queue_.load())
        init_record_queue();

    record_queue_.load()->reset();

    frame_record_worker_controller_.set_callback(callback);
    frame_record_worker_controller_.set_error_callback(error_callback_);
    frame_record_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);

    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    frame_record_worker_controller_.start(all_settings, get_cuda_streams().recorder_stream, record_queue_);
}

void Holovibes::stop_frame_record() { frame_record_worker_controller_.stop(false); }

void Holovibes::start_chart_record(const std::function<void()>& callback)
{
    chart_record_worker_controller_.set_callback(callback);
    chart_record_worker_controller_.set_error_callback(error_callback_);
    chart_record_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);

    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    chart_record_worker_controller_.start(all_settings);
}

void Holovibes::stop_chart_record() { chart_record_worker_controller_.stop(); }

void Holovibes::start_information_display()
{
    info_worker_controller_.set_error_callback(error_callback_);
    info_worker_controller_.set_priority(THREAD_DISPLAY_PRIORITY);
    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    info_worker_controller_.start(all_settings);
}

void Holovibes::stop_information_display() { info_worker_controller_.stop(); }

void Holovibes::start_compute()
{
    LOG_FUNC();
    init_input_queue(API.input.get_input_fd(), API.input.get_input_buffer_size());

    if (!compute_pipe_.load())
    {
        init_record_queue();
        compute_pipe_.store(std::make_shared<Pipe>(*(input_queue_.load()),
                                                   *(record_queue_.load()),
                                                   get_cuda_streams().compute_stream,
                                                   realtime_settings_.settings_));
    }

    compute_worker_controller_.set_error_callback(error_callback_);
    compute_worker_controller_.set_priority(THREAD_COMPUTE_PRIORITY);
    compute_worker_controller_.start(compute_pipe_);
}

void Holovibes::stop_compute()
{
    frame_record_worker_controller_.stop();
    chart_record_worker_controller_.stop();
    compute_worker_controller_.stop();
}

} // namespace holovibes
