/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "worker.hh"
#include "queue.hh"
#include "chrono.hh"
//#include <chrono>

namespace holovibes::worker
{
/*! \class FrameReadWorker
 *
 * \brief Abstract class used to read frames
 */
class FrameReadWorker : public Worker
{
  public:
    FrameReadWorker(std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue);

    /*! \brief Constructor
     *
     * \param gpu_input_queue The input queue
     */
    FrameReadWorker(std::atomic<std::shared_ptr<Queue>>& gpu_input_queue);

    virtual ~FrameReadWorker(){};

  protected:
    void compute_fps();

    /*! \brief The queue in which the frames are stored */
    std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue_;

    /*! \brief The current fps */
    std::shared_ptr<std::atomic<unsigned int>> current_fps_;
    std::atomic<unsigned int> processed_frames_;

    /*! \brief Useful for Input fps value. */

    Chrono chrono_;

    float current_display_rate = 30.0f;
    float time_to_wait = 33.0f;

    const cudaStream_t stream_;
};
} // namespace holovibes::worker
