/*! \file
 *
 * \brief Defines the DisplayQueue class for managing display queues.
 */
#pragma once

#include "frame_desc.hh"

namespace holovibes
{
/*! \class DisplayQueue
 *
 * \brief Abstract base class for managing display queues.
 *
 * This class provides an interface for accessing the last image in the queue and managing
 * frame descriptors.
 */
class DisplayQueue
{
  public:
    /*! \brief Constructor
     *
     * \param fd Frame descriptor used to initialize the display queue.
     */
    DisplayQueue(const camera::FrameDescriptor& fd)
        : fd_(fd)
    {
    }

    /*! \brief Virtual destructor to ensure proper cleanup in derived classes */
    virtual ~DisplayQueue() = default;

    /*! \brief Retrieves the last image in the display queue.
     *
     * \return Pointer to the last image in the queue.
     */
    virtual void* get_last_image() const = 0;

    /*! \brief Gets the frame descriptor of the display queue.
     *
     * \return Reference to the frame descriptor.
     */
    const camera::FrameDescriptor& get_fd() const { return fd_; }

  protected:
    /*! \brief Sets the frame descriptor of the queue.
     *
     * \param fd Frame descriptor to be set.
     */
    void set_fd(const camera::FrameDescriptor& fd) { fd_ = fd; }

  protected:
    camera::FrameDescriptor fd_; /*!< Frame descriptor for the display queue */
};
} // namespace holovibes