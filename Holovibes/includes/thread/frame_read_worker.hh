/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "worker.hh"
#include "queue.hh"

namespace holovibes::worker
{
class FrameReadWorker : public Worker
{
  public:
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
    std::atomic<std::shared_ptr<Queue>>& gpu_input_queue_;

    std::atomic<unsigned int> processed_fps_;
};
} // namespace holovibes::worker