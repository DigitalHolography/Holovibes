#ifndef THREAD_CAPTURE_HH
# define THREAD_CAPTURE_HH

# include <thread>

# include <icamera.hh>
# include "queue.hh"

# include "ithread_input.hh"

namespace holovibes
{
	class ThreadCapture : public IThreadInput
  {
  public:
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

#endif /* !THREAD_CAPTURE_HH */