/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include "camera_frame_read_worker.hh"
#include "holovibes.hh"

namespace holovibes::worker
{
    CameraFrameReadWorker::CameraFrameReadWorker(std::shared_ptr<camera::ICamera> camera,
                                                std::atomic<std::shared_ptr<Queue>>& gpu_input_queue) :
        FrameReadWorker(gpu_input_queue),
        camera_(camera)
    {}


    void CameraFrameReadWorker::run()
    {
        const camera::FrameDescriptor& camera_fd = camera_->get_fd();

        // Update information container
        std::string input_format = std::to_string(camera_fd.width) + std::string("x")
                                   + std::to_string(camera_fd.height) + std::string(" - ")
                                   + std::to_string(camera_fd.depth * 8) + std::string("bits");

        InformationContainer& info = Holovibes::instance().get_info_container();
        info.add_indication(InformationContainer::IndicationType::IMG_SOURCE, camera_->get_name());
        info.add_indication(InformationContainer::IndicationType::INPUT_FORMAT, std::ref(input_format));
        info.add_processed_fps(InformationContainer::FpsType::INPUT_FPS, std::ref(processed_fps_));

        try
        {
            camera_->init_camera();
            camera_->start_acquisition();

            while (!stop_requested_)
            {
                auto frame = camera_->get_frame();
                gpu_input_queue_.load()->enqueue(frame, cudaMemcpyHostToDevice);
                processed_fps_ += 1;
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
