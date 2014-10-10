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
    ThreadCapture(camera::Camera& cam, Queue& q)
      : cam_(cam),
      q_(q),
      th_(&ThreadCapture::capture, this),
      is_on_(true)
    {
    }

    ~ThreadCapture()
    {
      is_on_ = false;
      th_.join();
    }

  private:
    camera::Camera& cam_;
    Queue& q_;
    std::thread th_;
    bool is_on_;

    void capture()
    {
      while (is_on_)
        q_.enqueue(cam_.get_frame());
    }
  };
}

#endif /* !THREAD_CAPTURE_HH */