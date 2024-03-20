/*! \file
 *
 * \brief #TODO Add a description for this file
 */
#pragma once

#include "frame_desc.hh"

namespace holovibes
{
/*! \class DisplayQueue
 *
 * \brief #TODO Add a description for this class
 */
class DisplayQueue
{
  public:
    DisplayQueue(const camera::FrameDescriptor& fd)
        : fd_(fd)
    {
    }

    virtual void* get_last_image() const = 0;

    const camera::FrameDescriptor& get_fd() const { return fd_; }

  protected:
    /*! \brief Sets the frame descriptor of the queue */
    void set_fd(const camera::FrameDescriptor& fd) { fd_ = fd; }

  protected:
    camera::FrameDescriptor fd_;
};
} // namespace holovibes
