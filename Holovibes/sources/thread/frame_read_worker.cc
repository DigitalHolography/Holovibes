#include "frame_read_worker.hh"
#include "chrono.hh"
#include "holovibes.hh"
#include "API.hh"

namespace holovibes::worker
{
FrameReadWorker::FrameReadWorker(std::atomic<std::shared_ptr<BatchInputQueue>>& input_queue)
    : Worker()
    , input_queue_(input_queue)
    , current_fps_(0)
    , stream_(Holovibes::instance().get_cuda_streams().reader_stream)
{
}
} // namespace holovibes::worker
