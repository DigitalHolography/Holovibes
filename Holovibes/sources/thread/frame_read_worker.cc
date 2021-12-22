#include "frame_read_worker.hh"
#include "holovibes.hh"
#include "API.hh"

namespace holovibes::worker
{
FrameReadWorker::FrameReadWorker(std::atomic<std::shared_ptr<BatchInputQueue>>& gpu_input_queue)
    : Worker()
    , gpu_input_queue_(gpu_input_queue)
    , current_fps_(0)
    , processed_frames_(0)
    , stream_(Holovibes::instance().get_cuda_streams().reader_stream)
{
}

void FrameReadWorker::compute_fps()
{
    auto tick = std::chrono::high_resolution_clock::now();
    auto waited_time = std::chrono::duration_cast<std::chrono::milliseconds>(tick - start_).count();

    if (waited_time > time_to_wait)
    {
        if (abs(api::get_display_rate() - current_display_rate) >= 0.001f)
        {
            current_display_rate = api::get_display_rate();
            time_to_wait = (1000 / (current_display_rate == 0 ? 1 : current_display_rate));
        }

        *current_fps_ = (processed_frames_ * (1000.f / waited_time));
        processed_frames_ = 0;
        start_ = std::chrono::high_resolution_clock::now();
    }
}
} // namespace holovibes::worker
