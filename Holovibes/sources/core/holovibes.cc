#include "holovibes.hh"
#include "queue.hh"

#include "API.hh"
#include "camera_dll.hh"
#include "tools.hh"
#include "holo_file.hh"
#include "icompute.hh"
#include "API.hh"

namespace holovibes
{

Holovibes& Holovibes::instance()
{
    static Holovibes instance;
    return instance;
}

void Holovibes::init_gpu_queues()
{
    LOG_DEBUG("init_gpu_queues");

    gpu_input_queue_.reset(
        new BatchInputQueue(api::get_input_buffer_size(), api::get_batch_size(), api::get_import_frame_descriptor()));

    gpu_output_queue_.reset(new Queue(api::detail::get_value<OutputFrameDescriptor>(),
                                      api::detail::get_value<OutputBufferSize>(),
                                      QueueType::OUTPUT_QUEUE));

    // FIXME API : This can be remove if holovibes still works
    FrameDescriptor output_fd = api::get_import_frame_descriptor();
    if (api::detail::get_value<ComputeMode>() == ComputeModeEnum::Hologram)
    {
        output_fd.depth = 2;
        if (api::detail::get_value<ImageType>() == ImageTypeEnum::Composite)
            output_fd.depth = 6;
    }

    if (output_fd != api::detail::get_value<OutputFrameDescriptor>())
        LOG_ERROR("Output frame descriptor ComputeModeEnum failed !!");
}

void Holovibes::destroy_gpu_queues()
{
    LOG_DEBUG("destroy_gpu_queues");
    gpu_input_queue_.reset(static_cast<BatchInputQueue*>(nullptr));
    gpu_output_queue_.reset(static_cast<Queue*>(nullptr));
}

void Holovibes::start_file_frame_read()
{
    LOG_DEBUG("start_file_frame_read");
    file_frame_read_worker_controller_.start();
}

void Holovibes::stop_file_frame_read()
{
    LOG_DEBUG("stop_file_frame_read");
    file_frame_read_worker_controller_.stop();
}

void Holovibes::start_camera_frame_read()
{
    LOG_DEBUG("start_camera_frame_read");

    try
    {
        camera_read_worker_controller_.start(active_camera_);
    }
    catch (std::exception& e)
    {
        LOG_ERROR("Error at camera frame read start worker. (Exception: {})", e.what());
        stop_camera_frame_read();
        throw;
    }
}

void Holovibes::stop_camera_frame_read()
{
    LOG_FUNC();
    camera_read_worker_controller_.stop();
    active_camera_.reset();
}

void Holovibes::start_frame_record()
{
    LOG_DEBUG("start_frame_record");

    if (api::detail::get_value<BatchSize>() > api::detail::get_value<RecordBufferSize>())
    {
        LOG_ERROR("[RECORDER] Batch size must be lower than record queue size");
        return;
    }

    if (api::detail::get_value<ExportScriptPath>() == "")
        frame_record_worker_controller_.start();
    else
        batch_gpib_worker_controller_.start();
}


void Holovibes::stop_frame_record()
{
    LOG_DEBUG("stop_frame_record");

    if (api::detail::get_value<ExportScriptPath>() == "")
        frame_record_worker_controller_.stop();
    else
        batch_gpib_worker_controller_.stop();
}

void Holovibes::start_chart_record()
{
    LOG_DEBUG("start_chart_record");
    chart_record_worker_controller_.start();
}

void Holovibes::stop_chart_record()
{
    LOG_DEBUG("stop_chart_record");
    chart_record_worker_controller_.stop();
}

void Holovibes::start_information_display()
{
    LOG_DEBUG("start_information_display");
    info_worker_controller_.start();
}

void Holovibes::stop_information_display()
{
    LOG_DEBUG("stop_information_display");
    info_worker_controller_.stop();
}

void Holovibes::create_pipe()
{
    LOG_DEBUG("create_pipe");
    compute_pipe_.reset(new Pipe(*gpu_input_queue_, *gpu_output_queue_, get_cuda_streams().compute_stream));
}

void Holovibes::sync_pipe()
{
    LOG_DEBUG("synch_pipe_on_start");
    compute_pipe_->first_sync();
}

void Holovibes::destroy_pipe()
{
    LOG_DEBUG("destroy_pipe");
    compute_pipe_.reset(static_cast<Pipe*>(nullptr));
}

void Holovibes::start_compute()
{
    LOG_DEBUG("start_compute");
    compute_worker_controller_.start();
}

void Holovibes::stop_compute()
{
    LOG_DEBUG("stop_compute");
    compute_worker_controller_.stop();
    // Can't do this because this function is called by the computer worker itself
    // while (compute_worker_controller_.is_running());
}

void Holovibes::reload_streams() { cuda_streams_.reload(); }
} // namespace holovibes
