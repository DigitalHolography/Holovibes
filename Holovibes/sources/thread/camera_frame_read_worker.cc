#include "camera_frame_read_worker.hh"
#include "holovibes.hh"
#include "API.hh"
#include "global_state_holder.hh"

namespace holovibes::worker
{
CameraFrameReadWorker::CameraFrameReadWorker(std::shared_ptr<camera::ICamera> camera)
    : FrameReadWorker()
    , camera_(camera)
{
    GSH::fast_updates_map<IndicationType>.create_entry(IndicationType::IMG_SOURCE) = camera_->get_name();
    GSH::fast_updates_map<IndicationType>.create_entry(IndicationType::INPUT_FORMAT) = "FIXME Camera Format";

    to_record_ = 0;
    auto& entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::READ);
    entry.recorded = &total_captured_frames_;
    entry.to_record = &to_record_;
}

CameraFrameReadWorker::~CameraFrameReadWorker()
{
    GSH::fast_updates_map<IndicationType>.remove_entry(IndicationType::IMG_SOURCE);
    GSH::fast_updates_map<IndicationType>.remove_entry(IndicationType::INPUT_FORMAT);
    GSH::fast_updates_map<ProgressType>.remove_entry(ProgressType::READ);
}

void CameraFrameReadWorker::run()
{
    const FrameDescriptor& camera_fd = camera_->get_fd();

    // Update information container
    std::string input_format = std::to_string(camera_fd.width) + std::string("x") + std::to_string(camera_fd.height) +
                               std::string(" - ") + std::to_string(camera_fd.depth * 8) + std::string("bit");
    GSH::fast_updates_map<IndicationType>.create_entry(IndicationType::INPUT_FORMAT, true) = input_format;

    try
    {
        camera_->start_acquisition();

        while (!stop_requested_)
        {
            auto captured_fd = camera_->get_frames();
            enqueue_loop(captured_fd, camera_fd);
        }

        api::get_gpu_input_queue().stop_producer();
        camera_->stop_acquisition();
        camera_->shutdown_camera();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("[CAPTURE] {}", e.what());
    }
}

void CameraFrameReadWorker::enqueue_loop(const camera::CapturedFramesDescriptor& captured_fd,
                                         const FrameDescriptor& camera_fd)
{
    auto copy_kind = captured_fd.on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    for (unsigned i = 0; i < captured_fd.count1; ++i)
    {
        auto ptr = reinterpret_cast<uint8_t*>(captured_fd.region1) + i * camera_fd.get_frame_size();
        api::get_gpu_input_queue().enqueue(ptr, copy_kind);
    }

    for (unsigned i = 0; i < captured_fd.count2; ++i)
    {
        auto ptr = reinterpret_cast<uint8_t*>(captured_fd.region2) + i * camera_fd.get_frame_size();
        api::get_gpu_input_queue().enqueue(ptr, copy_kind);
    }

    total_captured_frames_ += captured_fd.count1 + captured_fd.count2;
    processed_frames_ += captured_fd.count1 + captured_fd.count2;
    compute_fps();

    api::get_gpu_input_queue().sync_current_batch();
}

} // namespace holovibes::worker
