#include "frame_read_worker.hh"
#include "chrono.hh"
#include "holovibes.hh"
#include "API.hh"

namespace holovibes::worker
{
FrameReadWorker::FrameReadWorker()
    : Worker()
    , current_fps_(0)
    , processed_frames_(0)
    , stream_(Holovibes::instance().get_cuda_streams().reader_stream)
{
    GSH::fast_updates_map<FpsType>.create_entry(FpsType::INPUT_FPS) = &current_fps_;

    to_record_ = api::get_nb_frame_to_read();
    auto& entry = GSH::fast_updates_map<ProgressType>.create_entry(ProgressType::FILE_READ);
    entry.recorded = &processed_frames_;
    entry.to_record = &to_record_;
}

FrameReadWorker::~FrameReadWorker() { GSH::fast_updates_map<FpsType>.remove_entry(FpsType::INPUT_FPS); }

void FrameReadWorker::compute_fps()
{
    chrono_.stop();
    auto waited_time = chrono_.get_milliseconds();

    if (waited_time > time_to_wait)
    {
        if (abs(api::get_display_rate() - current_display_rate) >= 0.001f)
        {
            current_display_rate = api::get_display_rate();
            time_to_wait = (1000 / (current_display_rate == 0 ? 1 : current_display_rate));
        }

        current_fps_ = processed_frames_ * (1000.f / waited_time);
        processed_frames_ = 0;
        chrono_.start();
    }
}
} // namespace holovibes::worker
