/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"
#include "pipe.hh"

namespace holovibes
{
class ICompute;
class Queue;
class BatchInputQueue;
} // namespace holovibes

namespace holovibes::worker
{
/*! \class ComputeWorker
 *
 * \brief Class used for all ComputeModeEnums
 */
class ComputeWorker final : public Worker
{
  public:
    /*!
     * \param pipe The compute pipe used to perform all operations
     * \param input Input queue that is filled either by the file_frame_read_worker or the camera_frame_read_worker
     * \param output Output queue that store processed images for display
     */
    ComputeWorker();

    void stop() override;

    void run() override;

  private:
    const cudaStream_t stream_;
};
} // namespace holovibes::worker
