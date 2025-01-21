/*! \file
 *
 * \brief Worker used to perform all computations
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
 * \brief Class used for all computations
 */
class ComputeWorker final : public Worker
{
  public:
    /*!
     * \param pipe The compute pipe used to perform all operations
     * \param input Input queue that is filled either by the file_frame_read_worker or the camera_frame_read_worker
     */
    ComputeWorker(std::atomic<std::shared_ptr<Pipe>>& pipe);

    void stop() override;

    void run() override;

  private:
    /*! \brief The compute pipe used to perform all operations */
    std::atomic<std::shared_ptr<Pipe>>& pipe_;

    const cudaStream_t stream_;
};
} // namespace holovibes::worker
