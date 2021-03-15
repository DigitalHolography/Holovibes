/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "frame_read_worker.hh"

// Fast forward declaration
namespace camera
{
class ICamera;
}

namespace holovibes
{
namespace worker
{
/*!
 *  \brief    Class used to read frames from a camera
 */
class CameraFrameReadWorker : public FrameReadWorker
{
  public:
    /*!
     *  \brief    Constructor
     *
     *  \param    camera            The camera used
     *  \param    gpu_input_queue   The input queue
     */
    CameraFrameReadWorker(std::shared_ptr<camera::ICamera> camera,
                          std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue);

    void run() override;

  private:
    //! The camera giving the images
    std::shared_ptr<camera::ICamera> camera_;
};
} // namespace worker
} // namespace holovibes