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
    cudaXStreamSynchronize(batch_streams_[end_index_]);

    return data_ + (static_cast<size_t>(end_index_) * batch_size_ +
                    curr_batch_counter_) *
                       frame_size_;
}

inline bool BatchInputQueue::is_empty() const { return size_ == 0; }

inline uint BatchInputQueue::get_size() const { return size_; }

inline bool BatchInputQueue::has_overridden() const { return has_overridden_; }

inline const void* BatchInputQueue::get_data() const { return data_; }

inline const camera::FrameDescriptor& BatchInputQueue::get_fd() const
{
    return fd_;
}

inline uint BatchInputQueue::get_frame_size() const { return frame_size_; }

inline uint BatchInputQueue::get_frame_res() const { return frame_res_; }
} // namespace holovibes
