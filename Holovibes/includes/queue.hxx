/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "queue.hh"

namespace holovibes
{
inline size_t Queue::get_frame_size() const { return frame_size_; }

inline void* Queue::get_data() const { return data_; }

inline size_t Queue::get_frame_res() const { return frame_res_; }

inline unsigned int Queue::get_size() const { return size_; }

inline unsigned int Queue::get_max_size() const { return max_size_; }

inline void* Queue::get_start() const
{
    return data_.get() + start_index_ * frame_size_;
}

inline unsigned int Queue::get_start_index() const { return start_index_; }

inline void* Queue::get_end() const
{
    return data_.get() + ((start_index_ + size_) % max_size_) * frame_size_;
}

inline void* Queue::get_last_image() const
{
    MutexGuard mGuard(mutex_);
    // if the queue is empty, return a random frame
    return data_.get() + ((start_index_ + size_ - 1) % max_size_) * frame_size_;
}

inline unsigned int Queue::get_end_index() const
{
    return (start_index_ + size_) % max_size_;
}

inline std::mutex& Queue::get_guard() { return mutex_; }

inline bool Queue::has_overridden() const { return has_overridden_; }
} // namespace holovibes
