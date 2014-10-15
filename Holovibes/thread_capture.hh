#ifndef THREAD_CAPTURE_HH
# define THREAD_CAPTURE_HH

# include <thread>

# include "camera.hh"
# include "queue.hh"

namespace holovibes
{
  class ThreadCapture
  {
  public:
    ThreadCapture(camera::Camera& camera, unsigned int buffer_nb_elts);
    ~ThreadCapture();

    Queue& get_queue()
    {
      return queue_;
    }

  private:
    void thread_proc();

  private:
    camera::Camera& camera_;
    Queue queue_;
    bool stop_requested_;
    std::thread thread_;
  };
}

#endif /* !THREAD_CAPTURE_HH */