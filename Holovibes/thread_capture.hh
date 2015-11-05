#pragma once

# include <thread>
# include <icamera.hh>
# include "queue.hh"
# include "ithread_input.hh"

namespace holovibes
{
  /*! \brief Thead, add frame from camera to queue
   *
   * While thread is running, thread get_frame from camera
   * and enqueue then in queue
   */
  class ThreadCapture : public IThreadInput
  {
  public:
    /*! \param camera must be initialize */
    ThreadCapture(camera::ICamera& camera, Queue& input);
    ~ThreadCapture();

  private:
    void thread_proc();

  private:
    camera::ICamera& camera_;
    Queue& queue_;
    std::thread thread_;
  };
}