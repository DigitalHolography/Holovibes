#ifndef THREAD_COMPUTE_HH_
# define THREAD_COMPUTE_HH_

#include <thread>

namespace holovibes
{
  class ThreadCompute
  {
  public:
    ThreadCompute();
    ~ThreadCompute();

  private:
    bool compute_on_;
    std::thread thread_;
  };
}

#endif /* !THREAD_COMPUTE_HH_ */