#pragma once
#include <atomic>
#include <cstdint>
#include <vector>

// A simple SPSC ring buffer carrying frame ID + timestamps.
// Capacity must be a power of two.
struct FrameStamp
{
    uint64_t id;
    uint64_t synced_us; // Unix epoch microseconds
    uint64_t camera_us; // camera clock microseconds (0 if not available)
    uint64_t offset_us; // host-camera offset microseconds
};

class StampQueue
{
  public:
    explicit StampQueue(size_t capacity_pow2 = 1 << 20)
        : mask_(capacity_pow2 - 1)
        , buf_(capacity_pow2)
        , head_(0)
        , tail_(0)
    {
    }

    // Producer: push n consecutive stamps from parallel arrays
    inline void push_range_from_arrays(
        const uint64_t* ids, const uint64_t* synced, const uint64_t* camera, const uint64_t* offset, uint32_t n)
    {
        for (uint32_t i = 0; i < n; ++i)
        {
            // wait while full (leave one slot free)
            while (((head_.load(std::memory_order_relaxed) - tail_.load(std::memory_order_acquire)) & mask_) == mask_)
            {
            }
            const size_t idx = head_.load(std::memory_order_relaxed) & mask_;
            buf_[idx].id = ids[i];
            buf_[idx].synced_us = synced[i];
            buf_[idx].camera_us = camera ? camera[i] : 0;
            buf_[idx].offset_us = offset ? offset[i] : 0;
            head_.store(head_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
        }
    }

    inline FrameStamp pop_one_blocking()
    {
        while (tail_.load(std::memory_order_relaxed) == head_.load(std::memory_order_acquire))
        {
        }
        const size_t idx = tail_.load(std::memory_order_relaxed) & mask_;
        const FrameStamp v = buf_[idx];
        tail_.store(tail_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
        return v;
    }

    inline void clear()
    {
        const auto h = head_.load(std::memory_order_acquire);
        tail_.store(h, std::memory_order_release);
    }

  private:
    size_t mask_;
    std::vector<FrameStamp> buf_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};

extern StampQueue g_record_stamp_queue;
