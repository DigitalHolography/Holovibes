#include "camera_frame_read_worker.hh"
#include "holovibes.hh"

namespace holovibes::worker
{
CameraFrameReadWorker::CameraFrameReadWorker(std::shared_ptr<camera::ICamera> camera,
                                             std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue)
    : FrameReadWorker(gpu_input_queue)
    , camera_(camera)
{
}

void CameraFrameReadWorker::run()
{
    const camera::FrameDescriptor& camera_fd = camera_->get_fd();

    // Update information container
    std::string input_format = std::to_string(camera_fd.width) + std::string("x") + std::to_string(camera_fd.height) +
                               std::string(" - ") + std::to_string(camera_fd.depth * 8) + std::string("bit");

    InformationContainer& info = Holovibes::instance().get_info_container();
    info.add_indication(InformationContainer::IndicationType::IMG_SOURCE, camera_->get_name());
    info.add_indication(InformationContainer::IndicationType::INPUT_FORMAT, std::ref(input_format));
    info.add_processed_fps(InformationContainer::FpsType::INPUT_FPS, std::ref(processed_fps_));

    try
    {
        camera_->start_acquisition();

        while (!stop_requested_)
        {
            auto captured_fd = camera_->get_frames();
            enqueue_loop(captured_fd, camera_fd);
        }

        gpu_input_queue_.load()->stop_producer();
        camera_->stop_acquisition();
        camera_->shutdown_camera();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR << "[CAPTURE] " << e.what();
    }

    info.remove_indication(InformationContainer::IndicationType::IMG_SOURCE);
    info.remove_indication(InformationContainer::IndicationType::INPUT_FORMAT);
    info.remove_processed_fps(InformationContainer::FpsType::INPUT_FPS);

    camera_.reset();
}

void CameraFrameReadWorker::enqueue_loop(const camera::CapturedFramesDescriptor& captured_fd,
                                         const camera::FrameDescriptor& camera_fd)
{
    auto copy_kind = captured_fd.on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    for (unsigned i = 0; i < captured_fd.count1; ++i)
    {
        auto ptr = (uint8_t*)(captured_fd.region1) + i * camera_fd.frame_size();
        gpu_input_queue_.load()->enqueue(ptr, copy_kind);
    }

    for (unsigned i = 0; i < captured_fd.count2; ++i)
    {
        auto ptr = (uint8_t*)(captured_fd.region2) + i * camera_fd.frame_size();
        gpu_input_queue_.load()->enqueue(ptr, copy_kind);
    }

    processed_fps_ += captured_fd.count1 + captured_fd.count2;
    gpu_input_queue_.load()->sync_current_batch();
}

} // namespace holovibes::worker
