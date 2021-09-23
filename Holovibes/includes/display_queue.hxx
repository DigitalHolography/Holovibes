#pragma once

#include "display_queue.hh"

namespace holovibes
{
inline DisplayQueue::DisplayQueue(const camera::FrameDescriptor& fd)
    : fd_(fd)
{
}

inline const camera::FrameDescriptor& DisplayQueue::get_fd() const { return fd_; }
} // namespace holovibes