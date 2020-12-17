/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "worker.hh"

#include "cuda_stream_handler.hh"

namespace holovibes
{
class ICompute;
class Queue;

namespace worker
{
class ComputeWorker : public Worker
{
  public:
    ComputeWorker(std::atomic<std::shared_ptr<ICompute>>& pipe,
                  std::atomic<std::shared_ptr<Queue>>& input,
                  std::atomic<std::shared_ptr<Queue>>& output);

    void stop() override;

    void run() override;

  private:
    std::atomic<std::shared_ptr<ICompute>>& pipe_;

    std::atomic<std::shared_ptr<Queue>>& input_;

    std::atomic<std::shared_ptr<Queue>>& output_;

    cuda_tools::CudaStreamHandler cuda_stream_handler_;
};
} // namespace worker
} // namespace holovibes