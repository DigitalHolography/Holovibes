#pragma once

#include "frame_desc.hh"

namespace holovibes
{
class DisplayQueue
{
  public:
    DisplayQueue(const camera::FrameDescriptor& fd);

    virtual void* get_last_image() const = 0;

    const camera::FrameDescriptor& get_fd() const;

  protected:
    camera::FrameDescriptor fd_;
};
} // namespace holovibes

#include "display_queue.hxx"