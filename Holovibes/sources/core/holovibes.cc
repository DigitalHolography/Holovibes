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

const float Holovibes::get_boundary()
{
    if (input_queue_.load())
    {
        FrameDescriptor fd = input_queue_.load()->get_fd();
        const float d = API.input.get_pixel_size() * 0.000001f;
        const float n = static_cast<float>(fd.height);
        return (n * d * d) / API.transform.get_lambda();
    }
    return 0.f;
}

bool Holovibes::is_recording() const { return frame_record_worker_controller_.is_running(); }

void Holovibes::init_input_queue(const unsigned int input_queue_size)
{
    if (!input_queue_.load())
        throw std::runtime_error("Must initialize input queue with a camera frame descriptor");

    init_input_queue(input_queue_.load()->get_fd(), input_queue_size);
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
    auto& api = API;
    auto device = api.record.get_record_queue_location();
    auto record_mode = api.record.get_record_mode();
    switch (record_mode)
    {
    case RecordMode::RAW:
    {
        if (!input_queue_.load())
        {
            LOG_DEBUG("Cannot create record queue : input queue not created");
            return;
        }
        LOG_DEBUG("RecordMode = Raw");
        if (!record_queue_.load())
            record_queue_ = std::make_shared<Queue>(input_queue_.load()->get_fd(),
                                                    api.record.get_record_buffer_size(),
                                                    QueueType::RECORD_QUEUE,
                                                    device);
        else
            record_queue_.load()->rebuild(input_queue_.load()->get_fd(),
                                          api.record.get_record_buffer_size(),
                                          get_cuda_streams().recorder_stream,
                                          device);

        LOG_DEBUG("Record queue allocated");
        break;
    }
    case RecordMode::HOLOGRAM:
    {
        LOG_DEBUG("RecordMode = Hologram");

        if (gpu_output_queue_.load() == nullptr)
        {
            api.compute.pipe_refresh();
            return;
        }

        auto record_fd = gpu_output_queue_.load()->get_fd();
        if (record_fd.depth == camera::PixelDepth::Bits8)
            record_fd.depth = camera::PixelDepth::Bits16;
        if (!record_queue_.load())
        {
            record_queue_ = std::make_shared<Queue>(record_fd,
                                                    api.record.get_record_buffer_size(),
                                                    QueueType::RECORD_QUEUE,
                                                    device);
        }
        else
            record_queue_.load()->rebuild(record_fd,
                                          api.record.get_record_buffer_size(),
                                          get_cuda_streams().recorder_stream,
                                          device);
        LOG_DEBUG("Record queue allocated");
        break;
    }
    case RecordMode::CUTS_YZ:
    case RecordMode::CUTS_XZ:
    {
        LOG_DEBUG("RecordMode = CUTS");
        camera::FrameDescriptor fd_xyz = gpu_output_queue_.load()->get_fd();
        fd_xyz.depth = camera::PixelDepth::Bits16; // Size of ushort
        if (record_mode == RecordMode::CUTS_XZ)
            fd_xyz.height = api.transform.get_time_transformation_size();
        else
            fd_xyz.width = api.transform.get_time_transformation_size();

        if (!record_queue_.load())
            record_queue_ =
                std::make_shared<Queue>(fd_xyz, api.record.get_record_buffer_size(), QueueType::RECORD_QUEUE, device);
        else
            record_queue_.load()->rebuild(fd_xyz,
                                          api.record.get_record_buffer_size(),
                                          get_cuda_streams().recorder_stream,
                                          device);
        LOG_DEBUG("Record queue allocated");
        break;
    }
    case RecordMode::MOMENTS:
    {
        LOG_DEBUG("RecordMode = Moments");
        camera::FrameDescriptor record_fd = input_queue_.load()->get_fd();
        record_fd.depth = camera::PixelDepth::Bits32;

        if (!record_queue_.load())
            record_queue_ = std::make_shared<Queue>(record_fd,
                                                    api.record.get_record_buffer_size(),
                                                    QueueType::RECORD_QUEUE,
                                                    device);
        else
            record_queue_.load()->rebuild(record_fd,
                                          api.record.get_record_buffer_size(),
                                          get_cuda_streams().recorder_stream,
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

void Holovibes::start_file_frame_read(const std::function<void()>& callback)
{
    CHECK(input_queue_.load() != nullptr);

    file_read_worker_controller_.set_callback(callback);
    file_read_worker_controller_.set_error_callback(error_callback_);
    file_read_worker_controller_.set_priority(THREAD_READER_PRIORITY);

    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    file_read_worker_controller_.start(input_queue_, all_settings);
}

void Holovibes::start_camera_frame_read(CameraKind camera_kind, const std::function<void()>& callback)
{
    try
    {
        const static std::map<CameraKind, LPCSTR> camera_dictionary = {
            {CameraKind::Adimec, "CameraAdimec.dll"},
            {CameraKind::BitflowCyton, "BitflowCyton.dll"},
            {CameraKind::IDS, "CameraIds.dll"},
            {CameraKind::Hamamatsu, "CameraHamamatsu.dll"},
            {CameraKind::xiQ, "CameraXiq.dll"},
            {CameraKind::xiB, "CameraXib.dll"},
            {CameraKind::OpenCV, "CameraOpenCV.dll"},
            {CameraKind::Phantom, "AmetekS710EuresysCoaxlinkOcto.dll"},
            {CameraKind::AmetekS711EuresysCoaxlinkQSFP, "AmetekS711EuresysCoaxlinkQsfp+.dll"},
            {CameraKind::AmetekS991EuresysCoaxlinkQSFP, "AmetekS991EuresysCoaxlinkQsfp+.dll"},
            {CameraKind::Ametek, "EuresyseGrabber.dll"},
            {CameraKind::Alvium, "CameraAlvium.dll"},
            {CameraKind::AutoDetectionPhantom, "CameraPhantomAutoDetection.dll"},
        };
        active_camera_ = camera::CameraDLL::load_camera(camera_dictionary.at(camera_kind));
    }
    catch (const std::exception& e)
    {
        LOG_INFO("Camera library cannot be loaded. (Exception: {})", e.what());
        throw;
    }

    try
    {
        auto& api = API;

        api.input.set_pixel_size(active_camera_->get_pixel_size());
        const camera::FrameDescriptor& camera_fd = active_camera_->get_fd();

        api.input.set_import_type(ImportType::Camera);
        api.input.set_input_fd(camera_fd);
        init_input_queue(camera_fd, api.input.get_input_buffer_size());

        camera_read_worker_controller_.set_callback(callback);
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
    active_camera_.reset();
    input_queue_.store(nullptr);
}

void Holovibes::start_frame_record(const std::function<void()>& callback)
{
    API.compute.pipe_refresh();
    if (get_setting<settings::BatchSize>().value > API.record.get_record_buffer_size())
    {
        LOG_ERROR("[RECORDER] Batch size must be lower than record queue size");
        return;
    }

    API.record.set_record_frame_count(get_setting<settings::RecordFrameCount>().value);

    if (!record_queue_.load())
        init_record_queue();

    frame_record_worker_controller_.set_callback(callback);
    frame_record_worker_controller_.set_error_callback(error_callback_);
    frame_record_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);

    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    frame_record_worker_controller_.start(all_settings, get_cuda_streams().recorder_stream, record_queue_);
}

void Holovibes::stop_frame_record() { frame_record_worker_controller_.stop(); }

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

void Holovibes::init_pipe()
{
    LOG_FUNC();
    camera::FrameDescriptor output_fd = input_queue_.load()->get_fd();
    if (API.compute.get_compute_mode() == Computation::Hologram)
    {
        output_fd.depth = camera::PixelDepth::Bits16;
        if (API.compute.get_img_type() == ImgType::Composite)
            output_fd.depth = camera::PixelDepth::Bits48;
    }
    gpu_output_queue_.store(std::make_shared<Queue>(output_fd,
                                                    static_cast<unsigned int>(API.compute.get_output_buffer_size()),
                                                    QueueType::OUTPUT_QUEUE));
    if (!compute_pipe_.load())
    {

        init_record_queue();
        compute_pipe_.store(std::make_shared<Pipe>(*(input_queue_.load()),
                                                   *(gpu_output_queue_.load()),
                                                   *(record_queue_.load()),
                                                   get_cuda_streams().compute_stream,
                                                   realtime_settings_.settings_));
    }
}

void Holovibes::start_compute_worker(const std::function<void()>& callback)
{
    compute_worker_controller_.set_callback(callback);
    compute_worker_controller_.set_error_callback(error_callback_);
    compute_worker_controller_.set_priority(THREAD_COMPUTE_PRIORITY);

    compute_worker_controller_.start(compute_pipe_, gpu_output_queue_);
}

void Holovibes::start_compute(const std::function<void()>& callback)
{
    /**
     * TODO change the assert by the // CHECK macro, but we don't know yet if it's a strict equivalent of it.
     * Here is a suggestion :
     * // CHECK(!!input_queue_.load()) << "Input queue not initialized";
     */
    CHECK(input_queue_.load() != nullptr, "Input queue not initialized");
    try
    {
        init_pipe();
    }
    catch (std::exception& e)
    {
        LOG_ERROR("Catch {}", e.what());
        return;
    }

    start_compute_worker(callback);

    while (!compute_pipe_.load())
        continue;
}

void Holovibes::stop_compute()
{
    frame_record_worker_controller_.stop();
    chart_record_worker_controller_.stop();
    compute_worker_controller_.stop();
}

void Holovibes::stop_all_worker_controller()
{
    info_worker_controller_.stop();
    stop_compute();
    stop_frame_read();
}

void Holovibes::reload_streams() { cuda_streams_.reload(); }
} // namespace holovibes
