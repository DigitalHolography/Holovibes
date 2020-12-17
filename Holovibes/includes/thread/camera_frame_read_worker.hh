/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "frame_read_worker.hh"

namespace camera
{
class ICamera;
}

namespace holovibes
{
namespace worker
{
class CameraFrameReadWorker : public FrameReadWorker
{
  public:
    CameraFrameReadWorker(std::shared_ptr<camera::ICamera> camera,
                          std::atomic<std::shared_ptr<Queue>>& gpu_input_queue);

    void run() override;

  private:
    std::shared_ptr<camera::ICamera> camera_;
};
} // namespace worker
} // namespace holovibes