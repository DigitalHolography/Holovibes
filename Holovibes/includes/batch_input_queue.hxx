/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "batch_input_queue.hh"

namespace holovibes
{
inline void* BatchInputQueue::get_last_image() const
{
    sync_current_batch();
    // Return the previous enqueued frame
    return data_.get() +
           ((start_index_ + curr_nb_frames_ - 1) % total_nb_frames_) *
               frame_size_;
}

inline uint BatchInputQueue::wait_and_lock(const std::atomic<uint>& index)
{
    uint tmp_index;
    while (true)
    {
        tmp_index = index.load();
        if (batch_mutexes_[tmp_index].try_lock())
            break;
    }
    return tmp_index;
}

inline bool BatchInputQueue::is_empty() const { return size_ == 0; }

inline uint BatchInputQueue::get_size() const { return size_; }

inline bool BatchInputQueue::has_overridden() const { return has_overridden_; }

inline const void* BatchInputQueue::get_data() const { return data_; }

inline uint BatchInputQueue::get_total_nb_frames() const
{
    return total_nb_frames_;
}

inline const camera::FrameDescriptor& BatchInputQueue::get_fd() const
{
    return fd_;
}

inline uint BatchInputQueue::get_frame_size() const { return frame_size_; }

inline uint BatchInputQueue::get_frame_res() const { return frame_res_; }
} // namespace holovibes
