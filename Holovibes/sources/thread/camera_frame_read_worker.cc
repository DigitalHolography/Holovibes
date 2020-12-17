/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "camera_frame_read_worker.hh"
#include "holovibes.hh"

namespace holovibes::worker
{
CameraFrameReadWorker::CameraFrameReadWorker(
    std::shared_ptr<camera::ICamera> camera,
    std::atomic<std::shared_ptr<Queue>>& gpu_input_queue)
    : FrameReadWorker(gpu_input_queue)
    , camera_(camera)
{
}

void CameraFrameReadWorker::run()
{
    const camera::FrameDescriptor& camera_fd = camera_->get_fd();

    // Update information container
    std::string input_format =
        std::to_string(camera_fd.width) + std::string("x") +
        std::to_string(camera_fd.height) + std::string(" - ") +
        std::to_string(camera_fd.depth * 8) + std::string("bits");

    InformationContainer& info = Holovibes::instance().get_info_container();
    info.add_indication(InformationContainer::IndicationType::IMG_SOURCE,
                        camera_->get_name());
    info.add_indication(InformationContainer::IndicationType::INPUT_FORMAT,
                        std::ref(input_format));
    info.add_processed_fps(InformationContainer::FpsType::INPUT_FPS,
                           std::ref(processed_fps_));

    try
    {
        camera_->start_acquisition();

        while (!stop_requested_)
        {
            camera::CapturedFramesDescriptor res = camera_->get_frames();
            gpu_input_queue_.load()->enqueue(
                res.data,
                stream_.get(),
                res.on_gpu ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice);
            processed_fps_ += res.count;
        }

        camera_->stop_acquisition();
        camera_->shutdown_camera();
    }
    catch (const std::exception& e)
    {
        LOG_ERROR("[CAPTURE] " + std::string(e.what()));
    }

    info.remove_indication(InformationContainer::IndicationType::IMG_SOURCE);
    info.remove_indication(InformationContainer::IndicationType::INPUT_FORMAT);
    info.remove_processed_fps(InformationContainer::FpsType::INPUT_FPS);

    camera_.reset();
}
} // namespace holovibes::worker
