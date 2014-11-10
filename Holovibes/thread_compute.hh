#ifndef THREAD_COMPUTE_HH_
# define THREAD_COMPUTE_HH_

# include <thread>
# include "queue.hh"
# include "compute_descriptor.hh"

namespace holovibes
{
  class ThreadCompute
  {
  public:
    ThreadCompute(
      const ComputeDescriptor& desc,
      Queue& q);
    ~ThreadCompute();

    Queue& get_queue();
  private:
    void thread_proc();

  private:
    ComputeDescriptor compute_desc_;
    Queue& input_q_;

    bool compute_on_;
    std::thread thread_;

    Queue* output_q_;
  };
}

#endif /* !THREAD_COMPUTE_HH_ */