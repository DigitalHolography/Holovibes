#ifndef THREAD_CAPTURE_HH
# define THREAD_CAPTURE_HH

# include <thread>

# include <icamera.hh>
# include "queue.hh"

namespace holovibes
{
  class ThreadCapture
  {
  public:
    ThreadCapture(camera::ICamera& camera, Queue& input);
    ~ThreadCapture();

  private:
    void thread_proc();

  private:
    camera::ICamera& camera_;
    Queue& queue_;
    bool stop_requested_;
    std::thread thread_;
  };
}

#endif /* !THREAD_CAPTURE_HH */