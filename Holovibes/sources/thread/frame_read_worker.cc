#include "frame_read_worker.hh"
#include "holovibes.hh"

namespace holovibes::worker
{
FrameReadWorker::FrameReadWorker(
    std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue)
    : Worker()
    , gpu_input_queue_(gpu_input_queue)
    , processed_fps_(0)
    , stream_(Holovibes::instance().get_cuda_streams().reader_stream)
{
}
} // namespace holovibes::worker
