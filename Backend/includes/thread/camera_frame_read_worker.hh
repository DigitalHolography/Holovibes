/*! \file
 *
 * \brief Declaration of the CameraFrameReadWorker class.
 */
#pragma once

#include "frame_read_worker.hh"
#include <atomic>

// Fast forward declaration
namespace camera
{
class ICamera;
struct CapturedFramesDescriptor;
} // namespace camera

namespace holovibes::worker
{
/*! \class CameraFrameReadWorker
 *
 * \brief Class used to read frames from a camera
 */
class CameraFrameReadWorker final : public FrameReadWorker
{
  public:
    /*! \brief Constructor
     *
     * \param camera The camera used
     * \param input_queue The input queue
     */
    CameraFrameReadWorker(std::shared_ptr<camera::ICamera> camera,
                          std::atomic<std::shared_ptr<BatchInputQueue>>& input_queue);

    void run() override;

  private:
    /*! \brief The camera giving the images */
    std::shared_ptr<camera::ICamera> camera_;
    std::shared_ptr<std::atomic<uint>> temperature_;

    void enqueue_loop(const camera::CapturedFramesDescriptor& captured_fd, const camera::FrameDescriptor& camera_fd);
};
} // namespace holovibes::worker
