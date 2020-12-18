/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "frame_read_worker.hh"
#include "holovibes.hh"

namespace holovibes::worker
{
FrameReadWorker::FrameReadWorker(
    std::atomic<std::shared_ptr<Queue>>& gpu_input_queue)
    : Worker()
    , gpu_input_queue_(gpu_input_queue)
    , processed_fps_(0)
    , stream_(Holovibes::instance().get_cuda_streams().reader_stream)
{
}
} // namespace holovibes::worker
