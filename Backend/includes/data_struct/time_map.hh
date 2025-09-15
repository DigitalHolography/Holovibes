// time_map.hh
#pragma once
#include <atomic>
#include <cstdint>
#include <vector>

/**
 * @class FrameTimeMap
 *
 * @brief Lock-free ring buffer that stores per-frame timing information.
 *
 * This class is used to associate frame IDs (monotonically increasing counters)
 * with their corresponding timestamps. It maintains three parallel rings:
 *  - Synced timestamp (in Unix µs): host-synchronized frame time
 *  - Camera timestamp (in µs since device boot, 0 if unavailable)
 *  - Offset (in µs): applied offset between host and camera clock domains
 *
 * The constructor takes a capacity that must be a power of two; this allows
 * efficient wrapping using a bit mask instead of modulo arithmetic.
 *
 * Typical usage:
 *  - The producer thread (camera acquisition) calls write_batch() once per
 *    batch of frames. This writes N consecutive entries starting at base_id.
 *  - Consumer threads (recorder, processing pipeline) call lookup_*() with
 *    a frame_id to retrieve the stored timing information.
 *
 * Concurrency model:
 *  - Single producer, multiple readers.
 *  - Writes and reads use relaxed atomics since there is only one writer,
 *    and eventual consistency is sufficient for time-stamping.
 */
class FrameTimeMap
{
  public:
    explicit FrameTimeMap(size_t capacity_pow2 = 1 << 20)
        : mask_(capacity_pow2 - 1)
        , ring_synced_(capacity_pow2)
        , ring_cam_(capacity_pow2)
        , ring_off_(capacity_pow2)
    {
    }

    inline void write_batch(uint64_t base_id,
                            uint64_t ts0_synced_us,
                            uint64_t period_us,
                            unsigned count,
                            uint64_t ts0_cam_us,
                            uint64_t offset_us)
    {
        for (unsigned i = 0; i < count; ++i)
        {
            const uint64_t id = (base_id + i) & mask_;
            const uint64_t synced = ts0_synced_us + uint64_t(i) * period_us;
            const uint64_t cam = ts0_cam_us ? (ts0_cam_us + uint64_t(i) * period_us) : 0;
            ring_synced_[id].store(synced, std::memory_order_relaxed);
            ring_cam_[id].store(cam, std::memory_order_relaxed);
            ring_off_[id].store(offset_us, std::memory_order_relaxed);
        }
    }

    inline uint64_t lookup_synced(uint64_t frame_id) const
    {
        return ring_synced_[frame_id & mask_].load(std::memory_order_relaxed);
    }
    inline uint64_t lookup_camera(uint64_t frame_id) const
    {
        return ring_cam_[frame_id & mask_].load(std::memory_order_relaxed);
    }
    inline uint64_t lookup_offset(uint64_t frame_id) const
    {
        return ring_off_[frame_id & mask_].load(std::memory_order_relaxed);
    }

  private:
    size_t mask_;
    std::vector<std::atomic<uint64_t>> ring_synced_;
    std::vector<std::atomic<uint64_t>> ring_cam_;
    std::vector<std::atomic<uint64_t>> ring_off_;
};