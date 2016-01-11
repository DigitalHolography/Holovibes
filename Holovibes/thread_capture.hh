/*! \file
 *
 * Thread encapsulation for obtaining images from a camera. */
#pragma once

# include <thread>

# include "ithread_input.hh"

/* Forward declarations. */
namespace holovibes
{
  class Queue;
}
namespace camera
{
  class ICamera;
}

namespace holovibes
{
  /*! \brief Thread encapsulation for obtaining images from a camera. */
  class ThreadCapture : public IThreadInput
  {
  public:
    /*! \brief Set a capture thread from a given camera and a destination queue.
     * \param The camera must be initialized
     * \param Destination queue */
    ThreadCapture(camera::ICamera& camera, Queue& input);

    ~ThreadCapture();

  private:
    /*! While the thread is running, the get_frame() function (see ICamera
     * interface) is called with the current camera. Images sent are enqueued. */
    void thread_proc();

  private:
    camera::ICamera& camera_;

    Queue& queue_;

    std::thread thread_;
  };
}