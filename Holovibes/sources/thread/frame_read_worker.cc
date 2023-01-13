#include "frame_read_worker.hh"
#include "chrono.hh"
#include "holovibes.hh"
#include "API.hh"

namespace holovibes::worker
{
FrameReadWorker::FrameReadWorker()
    : Worker()
    , current_fps_(0)
    , processed_frames_for_fps_(0)
    , stream_(Holovibes::instance().get_cuda_streams().reader_stream)
{
    GSH::fast_updates_map<FpsType>.create_entry(FpsType::INPUT_FPS) = &processed_frames_for_fps_;
}

FrameReadWorker::~FrameReadWorker() { GSH::fast_updates_map<FpsType>.remove_entry(FpsType::INPUT_FPS); }

} // namespace holovibes::worker
