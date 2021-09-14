/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"

namespace holovibes
{
class ICompute;
class Queue;
class BatchInputQueue;

namespace worker
{
/*! \class ComputeWorker
 *
 * \brief Class used for all computations
 */
class ComputeWorker : public Worker
{
  public:
    /*!
     * \param pipe The compute pipe used to perform all operations
     * \param input Input queue that is filled either by the
     * file_frame_read_worker or the camera_frame_read_worker
     * \param output Output queue that store processed images for display
     */
    ComputeWorker(std::atomic<std::shared_ptr<ICompute>>& pipe,
                  std::atomic<std::shared_ptr<BatchInputQueue>>& input,
                  std::atomic<std::shared_ptr<Queue>>& output);

    void stop() override;

    void run() override;

  private:
    //! The compute pipe used to perform all operations
    std::atomic<std::shared_ptr<ICompute>>& pipe_;

    //! Input queue that is filled either by the file_frame_read_worker or the
    //! camera_frame_read_worker
    std::atomic<std::shared_ptr<BatchInputQueue>>& input_;

    //! Output queue that store processed images for display
    std::atomic<std::shared_ptr<Queue>>& output_;

    const cudaStream_t stream_;
};
} // namespace worker
} // namespace holovibes
