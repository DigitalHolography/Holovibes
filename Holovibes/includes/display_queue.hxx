/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#pragma once

#include "display_queue.hh"

namespace holovibes
{
inline DisplayQueue::DisplayQueue(const camera::FrameDescriptor& fd)
    : fd_(fd)
{}

inline const camera::FrameDescriptor& DisplayQueue::get_fd() const
{
    return fd_;
}
} // namespace holovibes