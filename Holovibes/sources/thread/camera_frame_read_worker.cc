#include "camera_frame_read_worker.hh"
#include "holovibes.hh"
#include "global_state_holder.hh"

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

    auto entry1 = GSH::fast_updates_map<IndicationType>.create_entry(IndicationType::IMG_SOURCE, true);
    auto entry2 = GSH::fast_updates_map<IndicationType>.create_entry(IndicationType::INPUT_FORMAT, true);
    *entry1 = camera_->get_name();
    *entry2 = input_format;

    current_fps_ = GSH::fast_updates_map<FpsType>.create_entry(FpsType::INPUT_FPS);

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
        LOG_ERROR("[CAPTURE] {}", e.what());
    }

    GSH::fast_updates_map<IndicationType>.remove_entry(IndicationType::IMG_SOURCE);
    GSH::fast_updates_map<IndicationType>.remove_entry(IndicationType::INPUT_FORMAT);
    GSH::fast_updates_map<FpsType>.remove_entry(FpsType::INPUT_FPS);

    camera_.reset();
}

void CameraFrameReadWorker::enqueue_loop(const camera::CapturedFramesDescriptor& captured_fd,
                                         const camera::FrameDescriptor& camera_fd)
{
    auto copy_kind = captured_fd.on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    for (unsigned i = 0; i < captured_fd.count1; ++i)
    {
        auto ptr = (uint8_t*)(captured_fd.region1) + i * camera_fd.get_frame_size();
        gpu_input_queue_.load()->enqueue(ptr, copy_kind);
    }

    for (unsigned i = 0; i < captured_fd.count2; ++i)
    {
        auto ptr = (uint8_t*)(captured_fd.region2) + i * camera_fd.get_frame_size();
        gpu_input_queue_.load()->enqueue(ptr, copy_kind);
    }

    processed_frames_ += captured_fd.count1 + captured_fd.count2;
    compute_fps();

    gpu_input_queue_.load()->sync_current_batch();
}

} // namespace holovibes::worker
