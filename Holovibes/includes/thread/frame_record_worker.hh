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
class Queue;
class ICompute;

namespace worker
{
class FrameRecordWorker : public Worker
{
  public:
    FrameRecordWorker(const std::string& file_path,
                      std::optional<unsigned int> nb_frames_to_record,
                      bool raw_record,
                      bool square_output);

    void run() override;

  private:
    Queue& init_gpu_record_queue(std::shared_ptr<ICompute> pipe);

    void wait_for_frames(Queue& record_queue, std::shared_ptr<ICompute> pipe);

    void reset_gpu_record_queue(std::shared_ptr<ICompute> pipe);

    const std::string file_path_;

    std::optional<unsigned int> nb_frames_to_record_;

    std::atomic<unsigned int> processed_fps_;

    bool raw_record_;

    bool square_output_;

    const cudaStream_t stream_;
};
} // namespace worker
} // namespace holovibes
