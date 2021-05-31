#include "holovibes.hh"
#include "queue.hh"
#include "config.hh"
#include "camera_dll.hh"
#include "tools.hh"
#include "logger.hh"
#include "holo_file.hh"
#include "icompute.hh"

namespace holovibes
{
using camera::FrameDescriptor;

Holovibes& Holovibes::instance()
{
    static Holovibes instance;
    return instance;
}

void Holovibes::update_cd_for_cli(const unsigned int input_fps)
{
    // Compute time filter stride such as output fps = 20
    const unsigned int expected_output_fps = 20;
    cd_.time_transformation_stride =
        std::max(input_fps / expected_output_fps, static_cast<unsigned int>(1));
    cd_.batch_size = cd_.time_transformation_stride.load();

    // We force the contrast to not be enable in CLI mode
    cd_.contrast_enabled = false;
}

const float Holovibes::get_boundary()
{
    if (gpu_input_queue_.load())
    {
        FrameDescriptor fd = gpu_input_queue_.load()->get_fd();
        const float n = static_cast<float>(fd.height);
        const float d = cd_.pixel_size * 0.000001f;
        return (n * d * d) / cd_.lambda;
    }
    return 0.f;
}

void Holovibes::init_input_queue(const camera::FrameDescriptor& fd)
{
    camera::FrameDescriptor queue_fd = fd;

    gpu_input_queue_ = std::make_shared<BatchInputQueue>(
        global::global_config.input_queue_max_size,
        queue_fd);
}

void Holovibes::start_file_frame_read(const std::string& file_path,
                                      bool loop,
                                      unsigned int fps,
                                      unsigned int first_frame_id,
                                      unsigned int nb_frames_to_read,
                                      bool load_file_in_gpu,
                                      const std::function<void()>& callback)
{
    assert(gpu_input_queue_.load() != nullptr);

    file_read_worker_controller_.set_callback(callback);
    file_read_worker_controller_.set_priority(THREAD_READER_PRIORITY);
    file_read_worker_controller_.start(file_path,
                                       loop,
                                       fps,
                                       first_frame_id,
                                       nb_frames_to_read,
                                       load_file_in_gpu,
                                       gpu_input_queue_);
}

void Holovibes::start_camera_frame_read(CameraKind camera_kind,
                                        const std::function<void()>& callback)
{
    try
    {
        if (camera_kind == CameraKind::Adimec)
            active_camera_ = camera::CameraDLL::load_camera("CameraAdimec.dll");
        else if (camera_kind == CameraKind::IDS)
            active_camera_ = camera::CameraDLL::load_camera("CameraIds.dll");
        else if (camera_kind == CameraKind::Phantom)
            active_camera_ =
                camera::CameraDLL::load_camera("CameraPhantom.dll");
        else if (camera_kind == CameraKind::Hamamatsu)
            active_camera_ =
                camera::CameraDLL::load_camera("CameraHamamatsu.dll");
        else if (camera_kind == CameraKind::xiQ)
            active_camera_ = camera::CameraDLL::load_camera("CameraXiq.dll");
        else if (camera_kind == CameraKind::xiB)
            active_camera_ = camera::CameraDLL::load_camera("CameraXib.dll");
        else
            assert(!"Impossible case");

        cd_.pixel_size = active_camera_->get_pixel_size();
        const camera::FrameDescriptor& camera_fd = active_camera_->get_fd();

        init_input_queue(camera_fd);

        camera_read_worker_controller_.set_callback(callback);
        camera_read_worker_controller_.set_priority(THREAD_READER_PRIORITY);
        camera_read_worker_controller_.start(active_camera_, gpu_input_queue_);
    }
    catch (std::exception& e)
    {
        std::cout << e.what() << std::endl;
        stop_frame_read();
        throw;
    }
}

void Holovibes::stop_frame_read()
{
    camera_read_worker_controller_.stop();
    file_read_worker_controller_.stop();
    info_container_.clear();
    active_camera_.reset();
    gpu_input_queue_.store(nullptr);
}

void Holovibes::start_frame_record(
    const std::string& path,
    std::optional<unsigned int> nb_frames_to_record,
    bool raw_record,
    bool square_output,
    const std::function<void()>& callback)
{
    frame_record_worker_controller_.set_callback(callback);
    frame_record_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);
    frame_record_worker_controller_.start(path,
                                          nb_frames_to_record,
                                          raw_record,
                                          square_output);
}

void Holovibes::stop_frame_record() { frame_record_worker_controller_.stop(); }

void Holovibes::start_chart_record(const std::string& path,
                                   const unsigned int nb_points_to_record,
                                   const std::function<void()>& callback)
{
    chart_record_worker_controller_.set_callback(callback);
    chart_record_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);
    chart_record_worker_controller_.start(path, nb_points_to_record);
}

void Holovibes::stop_chart_record() { chart_record_worker_controller_.stop(); }

void Holovibes::start_batch_gpib(const std::string& batch_input_path,
                                 const std::string& output_path,
                                 unsigned int nb_frames_to_record,
                                 RecordMode record_mode,
                                 bool square_output,
                                 const std::function<void()>& callback)
{
    batch_gpib_worker_controller_.stop();
    batch_gpib_worker_controller_.set_callback(callback);
    batch_gpib_worker_controller_.set_priority(THREAD_RECORDER_PRIORITY);
    batch_gpib_worker_controller_.start(batch_input_path,
                                        output_path,
                                        nb_frames_to_record,
                                        record_mode,
                                        square_output);
}

void Holovibes::stop_batch_gpib() { batch_gpib_worker_controller_.stop(); }

void Holovibes::start_information_display(bool is_cli,
                                          const std::function<void()>& callback)
{
    info_worker_controller_.set_callback(callback);
    info_worker_controller_.set_priority(THREAD_DISPLAY_PRIORITY);
    info_worker_controller_.start(is_cli, info_container_);
}

void Holovibes::stop_information_display() { info_worker_controller_.stop(); }

void Holovibes::start_compute(const std::function<void()>& callback)
{
    assert(gpu_input_queue_.load() && "Input queue not initialized");

    compute_worker_controller_.set_callback(callback);
    compute_worker_controller_.set_priority(THREAD_COMPUTE_PRIORITY);
    compute_worker_controller_.start(compute_pipe_,
                                     gpu_input_queue_,
                                     gpu_output_queue_);

    while (!compute_pipe_.load())
        continue;
}

void Holovibes::stop_compute()
{
    frame_record_worker_controller_.stop();
    chart_record_worker_controller_.stop();
    batch_gpib_worker_controller_.stop();
    compute_worker_controller_.stop();
}

void Holovibes::stop_all_worker_controller()
{
    info_worker_controller_.stop();
    stop_compute();
    stop_frame_read();
}
} // namespace holovibes
