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
class BatchInputQueue;

namespace worker
{
class ComputeWorker : public Worker
{
  public:
    ComputeWorker(std::atomic<std::shared_ptr<ICompute>>& pipe,
                  std::atomic<std::shared_ptr<BatchInputQueue>>& input,
                  std::atomic<std::shared_ptr<Queue>>& output);

    void stop() override;

    void run() override;

  private:
    std::atomic<std::shared_ptr<ICompute>>& pipe_;

    std::atomic<std::shared_ptr<BatchInputQueue>>& input_;

    std::atomic<std::shared_ptr<Queue>>& output_;

    const cudaStream_t stream_;
};
} // namespace worker
} // namespace holovibes