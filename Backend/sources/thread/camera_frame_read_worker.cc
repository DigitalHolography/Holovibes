#include "camera_frame_read_worker.hh"
#include "holovibes.hh"
#include "fast_updates_holder.hh"
#include "api.hh"
#include "time_map.hh"
#include "id_queue.hh"

extern FrameTimeMap g_time_map;
extern IdQueue g_record_id_queue;

namespace holovibes::worker
{
CameraFrameReadWorker::CameraFrameReadWorker(std::shared_ptr<camera::ICamera> camera,
                                             std::atomic<std::shared_ptr<BatchInputQueue>>& input_queue)
    : FrameReadWorker(input_queue)
    , camera_(camera)
{
}

void CameraFrameReadWorker::run()
{
    const camera::FrameDescriptor& camera_fd = camera_->get_fd();

    // Update information container
    std::string input_format = std::to_string(camera_fd.width) + std::string("x") + std::to_string(camera_fd.height) +
                               std::string(" - ") + std::to_string(camera_fd.depth * 8) + std::string("bit");

    auto entry1 = FastUpdatesMap::map<IndicationType>.create_entry(IndicationType::IMG_SOURCE, true);
    auto entry2 = FastUpdatesMap::map<IndicationType>.create_entry(IndicationType::INPUT_FORMAT, true);
    *entry1 = camera_->get_name();
    *entry2 = input_format;

    current_fps_ = FastUpdatesMap::map<IntType>.create_entry(IntType::INPUT_FPS);
    temperature_ = FastUpdatesMap::map<IntType>.create_entry(IntType::TEMPERATURE, true);

    try
    {
        next_frame_id_.store(0, std::memory_order_relaxed);
        camera_->start_acquisition();
        while (!stop_requested_)
        {
            auto captured_fd = camera_->get_frames();
            enqueue_loop(captured_fd, camera_fd);
        }

        input_queue_.load()->stop_producer();
        camera_->stop_acquisition();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("[CAPTURE] {}", e.what());
    }

    FastUpdatesMap::map<IndicationType>.remove_entry(IndicationType::IMG_SOURCE);
    FastUpdatesMap::map<IndicationType>.remove_entry(IndicationType::INPUT_FORMAT);
    FastUpdatesMap::map<IntType>.remove_entry(IntType::INPUT_FPS);
    FastUpdatesMap::map<IntType>.remove_entry(IntType::TEMPERATURE);
}

void CameraFrameReadWorker::enqueue_loop(const camera::CapturedFramesDescriptor& captured_fd,
                                         const camera::FrameDescriptor& camera_fd)
{
    cudaMemcpyKind copy_kind = captured_fd.on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice;

    // total frames in this callback
    const unsigned total = captured_fd.count1 + captured_fd.count2;

    // assign a contiguous ID range for this batch
    const uint64_t base_id = next_frame_id_.fetch_add(total, std::memory_order_relaxed);

    // store timestamps in global time map
    g_time_map.write_batch(base_id,
                           captured_fd.first_frame_timestamp_us, // ts0_synced (Unix µs)
                           captured_fd.frame_period_us,
                           total,
                           captured_fd.camera_timestamp_us, // ts0_cam (boot µs) or 0 if not available with this camera
                           captured_fd.frame_offset_us      // offset applied (µs)
    );

    // enqueue region1 with IDs
    if (captured_fd.count1 > 0)
    {
        auto ptr1 = static_cast<uint8_t*>(captured_fd.region1);
        input_queue_.load()->enqueue_with_ids(ptr1, copy_kind, captured_fd.count1, base_id);
    }
    // enqueue region2 with IDs
    if (captured_fd.count2 > 0)
    {
        auto ptr2 = static_cast<uint8_t*>(captured_fd.region2);
        input_queue_.load()->enqueue_with_ids(ptr2, copy_kind, captured_fd.count2, base_id + captured_fd.count1);
    }

    *current_fps_ += captured_fd.count1 + captured_fd.count2;
    *temperature_ = camera_->get_temperature();
    input_queue_.load()->sync_current_batch();
}

} // namespace holovibes::worker
