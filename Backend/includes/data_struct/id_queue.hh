#pragma once
#include <atomic>
#include <cstdint>
#include <vector>
#include "logger.hh"

class IdQueue
{
  public:
    explicit IdQueue(size_t capacity_pow2 = 1 << 20)
        : mask_(capacity_pow2 - 1)
        , buf_(capacity_pow2)
        , head_(0)
        , tail_(0)
    {
    }
    // Producer (camera) : push a range of IDs [base_id, base_id+n-1]
    inline void push_range(uint64_t base_id, uint32_t n)
    {
        for (uint32_t i = 0; i < n; ++i)
        {
            while (((head_.load(std::memory_order_relaxed) - tail_.load(std::memory_order_acquire)) & mask_) == mask_)
            {
            }
            size_t idx = head_.load(std::memory_order_relaxed) & mask_;
            buf_[idx].store(base_id + i, std::memory_order_relaxed);
            head_.store(head_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
        }
    }

    // Consumer (recorder) : pop 1 ID
    inline uint64_t pop_one_blocking()
    {
        while (tail_.load(std::memory_order_relaxed) == head_.load(std::memory_order_acquire))
        {
        }
        size_t idx = tail_.load(std::memory_order_relaxed) & mask_;
        uint64_t v = buf_[idx].load(std::memory_order_relaxed);
        tail_.store(tail_.load(std::memory_order_relaxed) + 1, std::memory_order_release);
        return v;
    }

    inline void clear()
    {
        // SPSC : suffisant de réaligner tail sur head
        auto h = head_.load(std::memory_order_acquire);
        tail_.store(h, std::memory_order_release);
    }

  private:
    size_t mask_;
    std::vector<std::atomic<uint64_t>> buf_;
    std::atomic<size_t> head_;
    std::atomic<size_t> tail_;
};

extern IdQueue g_record_id_queue;