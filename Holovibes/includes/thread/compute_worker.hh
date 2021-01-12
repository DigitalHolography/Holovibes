/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "worker.hh"

namespace holovibes
{
class ICompute;
class Queue;

namespace worker
{
/*!
 *  \brief Class used for all computations
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
                  std::atomic<std::shared_ptr<Queue>>& input,
                  std::atomic<std::shared_ptr<Queue>>& output);

    void stop() override;

    void run() override;

  private:
    //! The compute pipe used to perform all operations
    std::atomic<std::shared_ptr<ICompute>>& pipe_;

    //! Input queue that is filled either by the file_frame_read_worker or the
    //! camera_frame_read_worker
    std::atomic<std::shared_ptr<Queue>>& input_;

    //! Output queue that store processed images for display
    std::atomic<std::shared_ptr<Queue>>& output_;
};
} // namespace worker
} // namespace holovibes