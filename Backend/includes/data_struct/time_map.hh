// time_map.hh
#pragma once
#include <atomic>
#include <cstdint>
#include <vector>

class FrameTimeMap
{
  public:
    explicit FrameTimeMap(size_t capacity_pow2 = 1 << 20)
        : mask_(capacity_pow2 - 1)
        , ring_(capacity_pow2)
    {
    }

    // Write exact per-frame timestamps for a batch:
    inline void write_batch(uint64_t base_id, uint64_t ts0_us, uint64_t period_us, unsigned count)
    {
        for (unsigned i = 0; i < count; ++i)
        {
            ring_[(base_id + i) & mask_].store(ts0_us + uint64_t(i) * period_us, std::memory_order_relaxed);
        }
    }

    // Lookup (relaxed is fine; recorder is a consumer)
    inline uint64_t lookup(uint64_t frame_id) const { return ring_[(frame_id)&mask_].load(std::memory_order_relaxed); }

  private:
    size_t mask_;
    std::vector<std::atomic<uint64_t>> ring_;
};
