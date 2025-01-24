/*! \file display_queue.hh
 *
 * \brief Defines the DisplayQueue interface. It's used by the UI to fetch the last image in the queue.
 */
#pragma once

#include "frame_desc.hh"

namespace holovibes
{
/*! \class DisplayQueue
 *
 * \brief An interface used by the UI to fetch the last image in the queue.
 */
class DisplayQueue
{
  public:
    DisplayQueue(const camera::FrameDescriptor& fd)
        : fd_(fd)
    {
    }

    /*! \brief Returns the last image in the Queue of nullptr if the Queue is empty
     *
     * \return void* The last image of size fd.frame_res or nullptr if no frame
     */
    virtual void* get_last_image() const = 0;

    /*! \brief Returns the frame descriptor of the queue
     *
     * \return camera::FrameDescriptor& The frame descriptor
     */
    const camera::FrameDescriptor& get_fd() const { return fd_; }

  protected:
    /*! \brief Sets the frame descriptor of the queue
     *
     * \param[in] fd The frame descriptor
     */
    void set_fd(const camera::FrameDescriptor& fd) { fd_ = fd; }

  protected:
    camera::FrameDescriptor fd_;
};
} // namespace holovibes
