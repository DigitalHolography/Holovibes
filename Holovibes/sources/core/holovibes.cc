#include "holovibes.hh"
#include "queue.hh"

#include "API.hh"
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
    if (gpu_input_queue_.load())
    {
        FrameDescriptor fd = gpu_input_queue_.load()->get_fd();
        const float d = api::get_pixel_size() * 0.000001f;
        const float n = static_cast<float>(fd.height);
        return (n * d * d) / api::get_lambda();
    }
    return 0.f;
}

bool Holovibes::is_recording() const { return frame_record_worker_controller_.is_running(); }

void Holovibes::init_input_queue(const camera::FrameDescriptor& fd, const unsigned int input_queue_size)
{
    camera::FrameDescriptor queue_fd = fd;

    gpu_input_queue_ = std::make_shared<BatchInputQueue>(input_queue_size, get_setting<settings::BatchSize>().value, queue_fd);
}

// TODO(julesguillou): Why using input fps here?
void Holovibes::start_file_frame_read(const std::function<void()>& callback)
{
    CHECK(gpu_input_queue_.load() != nullptr);

    file_read_worker_controller_.set_callback(callback);
    file_read_worker_controller_.set_error_callback(error_callback_);
    file_read_worker_controller_.set_priority(THREAD_READER_PRIORITY);

    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    file_read_worker_controller_.start(gpu_input_queue_, all_settings);
}

void Holovibes::start_camera_frame_read(CameraKind camera_kind, const std::function<void()>& callback)
{
    try
    {
        const static std::map<CameraKind, LPCSTR> camera_dictionary = {
            {CameraKind::Adimec, "CameraAdimec.dll"},
            {CameraKind::BitflowCyton, "BitflowCyton.dll"},
            {CameraKind::IDS, "CameraIds.dll"},
            {CameraKind::Phantom, "CameraPhantom.dll"},
            {CameraKind::Hamamatsu, "CameraHamamatsu.dll"},
            {CameraKind::xiQ, "CameraXiq.dll"},
            {CameraKind::xiB, "CameraXib.dll"},
            {CameraKind::OpenCV, "CameraOpenCV.dll"},
        };
        active_camera_ = camera::CameraDLL::load_camera(camera_dictionary.at(camera_kind));
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("Camera library cannot be loaded. (Exception: {})", e.what());
        throw;
    }

    try
    {
        api::set_pixel_size(active_camera_->get_pixel_size());
        const camera::FrameDescriptor& camera_fd = active_camera_->get_fd();

        UserInterfaceDescriptor::instance().import_type_ = ImportType::Camera;
        init_input_queue(camera_fd, api::get_input_buffer_size());

        camera_read_worker_controller_.set_callback(callback);
        camera_read_worker_controller_.set_error_callback(error_callback_);
        camera_read_worker_controller_.set_priority(THREAD_READER_PRIORITY);
        camera_read_worker_controller_.start(active_camera_, gpu_input_queue_);
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
    gpu_input_queue_.store(nullptr);
}
/*
void Holovibes::start_cli_record_and_compute(const std::string& path,
                                             std::optional<unsigned int> nb_frames_to_record,
                                             RecordMode record_mode,
                                             unsigned int nb_frames_skip)
{
    start_frame_record(path, nb_frames_to_record, false, nb_frames_skip);
*/
void Holovibes::start_frame_record(const std::function<void()>& callback)
{
    api::pipe_refresh();
    if (get_setting<settings::BatchSize>().value > api::get_record_buffer_size())
    {
        LOG_ERROR("[RECORDER] Batch size must be lower than record queue size");
        return;
    }

    GSH::instance().set_nb_frames_to_record(nb_frames_to_record);

    frame_record_worker_controller_.set_callback(callback);
    frame_record_worker_controller_.set_error_callback(error_callback_);
    frame_record_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);

    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    frame_record_worker_controller_.start(all_settings, get_cuda_streams().recorder_stream);
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

void Holovibes::start_information_display(const std::function<void()>& callback)
{
    info_worker_controller_.set_callback(callback);
    info_worker_controller_.set_error_callback(error_callback_);
    info_worker_controller_.set_priority(THREAD_DISPLAY_PRIORITY);
    auto all_settings = std::tuple_cat(realtime_settings_.settings_);
    info_worker_controller_.start(all_settings);
}

void Holovibes::stop_information_display() { info_worker_controller_.stop(); }

void Holovibes::init_pipe()
{
    LOG_FUNC();
    camera::FrameDescriptor output_fd = gpu_input_queue_.load()->get_fd();
    if (get_setting<settings::ComputeMode>().value == Computation::Hologram)
    {
        output_fd.depth = 2;
        if (api::get_img_type() == ImgType::Composite)
            output_fd.depth = 6;
    }
    gpu_output_queue_.store(
        std::make_shared<Queue>(output_fd, get_setting<settings::OutputBufferSize>().value, QueueType::OUTPUT_QUEUE));
    if (!compute_pipe_.load())
    {
        compute_pipe_.store(std::make_shared<Pipe>(*(gpu_input_queue_.load()),
                                                   *(gpu_output_queue_.load()),
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
     * // CHECK(!!gpu_input_queue_.load()) << "Input queue not initialized";
     */
    CHECK(gpu_input_queue_.load() != nullptr, "Input queue not initialized");
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
