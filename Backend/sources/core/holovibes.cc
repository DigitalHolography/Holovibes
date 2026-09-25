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

bool Holovibes::is_recording() const
{
    if (chart_record_worker_controller_.is_running())
        return true;

    std::lock_guard lock(record_mutex_);
    const int active_index = active_recording_index_.load(std::memory_order_acquire);
    if (active_index < 0)
        return false;

    const auto& context = recording_contexts_[active_index];
    if (record_queues_[1])
        return context && context->acquiring.load(std::memory_order_acquire);

    return frame_record_worker_controllers_[active_index].is_running();
}

bool Holovibes::can_start_frame_record() const
{
    std::lock_guard lock(record_mutex_);
    if (is_recording())
        return false;

    if (!record_queue_.load())
        return true;

    const size_t queue_count = record_queues_[1] ? RECORD_QUEUE_COUNT : 1;
    for (size_t index = 0; index < queue_count; ++index)
        if (record_queues_[index] && !frame_record_worker_controllers_[index].is_running())
            return true;

    return false;
}

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
    {
        std::lock_guard lock(record_mutex_);
        for (auto& context : recording_contexts_)
        {
            if (!context)
                continue;
            context->acquiring.store(false, std::memory_order_release);
            std::get<2>(*context->progress) = std::get<0>(*context->progress).load();
        }
    }
    for (auto& controller : frame_record_worker_controllers_)
        controller.stop();

    std::lock_guard lock(record_mutex_);
    active_recording_.store(nullptr, std::memory_order_release);
    active_recording_index_.store(-1, std::memory_order_release);
    recording_contexts_.fill(nullptr);

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
    case RecordMode::OCT_CUBE:
    {
        fd.depth = camera::PixelDepth::Complex; // float2
        break;
    }
    case RecordMode::OCT_CUBE_FLOAT:
    {
        fd.depth = camera::PixelDepth::Bits32; // float
        break;
    }
    default:
    {
        LOG_DEBUG("RecordMode = None");
        return;
    }
    }

    static constexpr std::array queue_types = {
        QueueType::RECORD_QUEUE, QueueType::RECORD_QUEUE_2, QueueType::RECORD_QUEUE_3};
    const size_t queue_count = api.record.get_record_queue_multibuffering_enabled() ? RECORD_QUEUE_COUNT : 1;

    for (size_t index = 0; index < queue_count; ++index)
    {
        if (!record_queues_[index])
        {
            auto queue = std::make_shared<Queue>(fd, size, queue_types[index], device);
            // Queue may lower the configured size and recursively rebuild all queues when memory is tight.
            if (api.record.get_record_buffer_size() != size)
                return;
            record_queues_[index] = std::move(queue);
        }
        else
            record_queues_[index]->rebuild(fd, size, get_cuda_streams().recorder_stream, device);
    }
    for (size_t index = queue_count; index < RECORD_QUEUE_COUNT; ++index)
        record_queues_[index].reset();

    record_queue_.store(record_queues_[0]);

    LOG_DEBUG("Record queue allocated");
}

void Holovibes::start_file_frame_read()
{
    CHECK(input_queue_.load() != nullptr);

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

bool Holovibes::start_frame_record(const std::function<void()>& callback)
{
    std::lock_guard lock(record_mutex_);

    if (!record_queue_.load())
        init_record_queue();

    const size_t queue_count = record_queues_[1] ? RECORD_QUEUE_COUNT : 1;
    size_t index = queue_count;
    for (size_t candidate = 0; candidate < queue_count; ++candidate)
    {
        if (!frame_record_worker_controllers_[candidate].is_running())
        {
            index = candidate;
            break;
        }
    }
    if (index == queue_count)
    {
        LOG_WARN("No record queue is available; wait for a pending file save to finish");
        return false;
    }

    auto progress = FastUpdatesMap::map<RecordType>.get_entry(RecordType::FRAME);
    auto context = std::make_shared<RecordingContext>(record_queues_[index], progress);
    // Publish the replacement before releasing the previous context stored in this slot.
    active_recording_.store(context.get(), std::memory_order_release);
    active_recording_index_.store(static_cast<int>(index), std::memory_order_release);
    recording_contexts_[index] = context;
    record_queue_.store(context->queue);

    context->queue->reset();
    if (input_queue_.load())
        input_queue_.load()->reset_override();

    const bool multibuffering_enabled = queue_count > 1;
    auto& controller = frame_record_worker_controllers_[index];
    controller.set_callback(multibuffering_enabled ? std::function<void()>{} : callback);
    controller.set_error_callback(error_callback_);
    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    controller.start(all_settings,
                     get_cuda_streams().recorder_stream,
                     context,
                     multibuffering_enabled ? callback : std::function<void()>{});
    controller.set_priority(THREAD_RECORDER_PRIORITY);
    return true;
}

void Holovibes::stop_frame_record()
{
    std::lock_guard lock(record_mutex_);
    const int active_index = active_recording_index_.load(std::memory_order_acquire);
    if (active_index >= 0)
        frame_record_worker_controllers_[active_index].stop(false);
}

void Holovibes::finish_frame_acquisition(RecordingContext* recording)
{
    std::lock_guard lock(record_mutex_);
    recording->acquiring.store(false, std::memory_order_release);
    if (active_recording_.load(std::memory_order_acquire) == recording)
        API.record.set_frame_acquisition_enabled(false);
}

void Holovibes::start_chart_record(const std::function<void()>& callback)
{
    chart_record_worker_controller_.set_callback(callback);
    chart_record_worker_controller_.set_error_callback(error_callback_);
    chart_record_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);

    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    chart_record_worker_controller_.start(all_settings);
}

void Holovibes::stop_chart_record() { chart_record_worker_controller_.stop(); }

void Holovibes::start_benchmark()
{
    benchmark_worker_controller_.set_error_callback(error_callback_);
    benchmark_worker_controller_.set_priority(THREAD_DISPLAY_PRIORITY);
    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    benchmark_worker_controller_.start(all_settings);
}

void Holovibes::stop_benchmark() { benchmark_worker_controller_.stop(); }

void Holovibes::start_compute()
{
    LOG_FUNC();
    init_input_queue(API.input.get_input_fd(), API.input.get_input_buffer_size());

    if (!compute_pipe_.load())
    {
        // CLI recording starts before computation so no input frame can be missed. In that case the record queue
        // already belongs to the active RecordingContext and rebuilding it would stop the newly started worker.
        if (!active_recording_.load(std::memory_order_acquire))
            init_record_queue();
        compute_pipe_.store(std::make_shared<Pipe>(*(input_queue_.load()),
                                                   active_recording_,
                                                   get_cuda_streams().compute_stream,
                                                   realtime_settings_.settings_));
    }

    compute_worker_controller_.set_error_callback(error_callback_);
    compute_worker_controller_.set_priority(THREAD_COMPUTE_PRIORITY);
    compute_worker_controller_.start(compute_pipe_);
}

void Holovibes::stop_compute()
{
    {
        std::lock_guard lock(record_mutex_);
        for (auto& context : recording_contexts_)
        {
            if (!context)
                continue;
            context->acquiring.store(false, std::memory_order_release);
            std::get<2>(*context->progress) = std::get<0>(*context->progress).load();
        }
    }
    for (auto& controller : frame_record_worker_controllers_)
        controller.stop();
    active_recording_.store(nullptr, std::memory_order_release);
    active_recording_index_.store(-1, std::memory_order_release);
    chart_record_worker_controller_.stop();
    compute_worker_controller_.stop();
}

} // namespace holovibes
