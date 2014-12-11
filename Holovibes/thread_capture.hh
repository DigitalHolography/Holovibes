#ifndef THREAD_CAPTURE_HH
# define THREAD_CAPTURE_HH

# include <thread>

# include "camera/camera.hh"
# include "queue.hh"

namespace holovibes
{
  class ThreadCapture
  {
  public:
    ThreadCapture(camera::Camera& camera, Queue& input);
    ~ThreadCapture();

  private:
    void thread_proc();

  private:
    camera::Camera& camera_;
    Queue& queue_;
    bool stop_requested_;
    std::thread thread_;
  };
}

#endif /* !THREAD_CAPTURE_HH */