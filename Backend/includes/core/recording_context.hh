/*! \file recording_context.hh
 *
 * \brief State owned by one frame recording.
 */
#pragma once

#include <atomic>
#include <memory>

#include "fast_updates_holder.hh"
#include "queue.hh"
#include "stamp_queue.hh"

namespace holovibes
{
/*! \brief Keeps queue-local state together while a recording is acquired and saved.
 *
 * A context remains alive until its writer finishes. This is important when the
 * next recording starts on another queue: progress and frame timestamps must not
 * be read from the new recording's global state.
 */
struct RecordingContext
{
    RecordingContext(std::shared_ptr<Queue> record_queue, FastUpdatesHolder<RecordType>::Value record_progress)
        : queue(std::move(record_queue))
        , stamps(stamp_capacity(queue->get_max_size()))
        , progress(std::move(record_progress))
    {
    }

    static size_t stamp_capacity(size_t queue_capacity)
    {
        size_t capacity = 1;
        while (capacity <= queue_capacity)
            capacity <<= 1;
        return capacity;
    }

    std::shared_ptr<Queue> queue;
    StampQueue stamps;
    FastUpdatesHolder<RecordType>::Value progress;
    std::atomic<bool> acquiring{true};
    std::atomic<bool> input_overwritten{false};
};
} // namespace holovibes
