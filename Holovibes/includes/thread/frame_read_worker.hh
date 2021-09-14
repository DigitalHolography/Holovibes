/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"
#include "queue.hh"

namespace holovibes::worker
{
/*! \class FrameReadWorker
 *
 * \brief Abstract class used to read frames
 */
class FrameReadWorker : public Worker
{
  public:
    FrameReadWorker(
        std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue);

    /*!
     *  \brief    Constructor
     *
     *  \param    gpu_input_queue   The input queue
     */
    FrameReadWorker(std::atomic<std::shared_ptr<Queue>>& gpu_input_queue);

    /*!
     *  \brief    Default copy constructor
     */
    FrameReadWorker(const FrameReadWorker&) = default;

    /*!
     *  \brief    Default copy operator
     */
    FrameReadWorker& operator=(const FrameReadWorker&) = default;

  protected:
    //! The queue in which the frames are stored
    std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue_;

    //! The current fps
    std::atomic<unsigned int> processed_fps_;

    const cudaStream_t stream_;
};
} // namespace holovibes::worker
