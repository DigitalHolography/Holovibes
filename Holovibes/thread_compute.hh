#ifndef THREAD_COMPUTE_HH_
# define THREAD_COMPUTE_HH_

# include <thread>
# include "queue.hh"
# include "compute_descriptor.hh"
# include "pipeline.hh"

namespace holovibes
{
  class ThreadCompute
  {
  public:
    ThreadCompute(
      ComputeDescriptor& desc,
      Queue& input,
      Queue& output);
    ~ThreadCompute();

    Pipeline& get_pipeline()
    {
      return *pipeline_;
    }
  private:
    void thread_proc();

  private:
    ComputeDescriptor& compute_desc_;
    Queue& input_;
    Queue& output_;
    Pipeline* pipeline_;

    bool compute_on_;
    std::thread thread_;
  };
}

#endif /* !THREAD_COMPUTE_HH_ */