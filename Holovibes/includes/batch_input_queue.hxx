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
inline bool BatchInputQueue::is_empty() const { return size_ == 0; }

inline uint BatchInputQueue::get_size() const { return size_; }

inline bool BatchInputQueue::has_overridden() const { return has_overridden_; }

inline const void* BatchInputQueue::get_data() const { return data_; }

inline uint BatchInputQueue::get_frame_size() const { return frame_size_; }
} // namespace holovibes
