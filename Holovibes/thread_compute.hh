#ifndef THREAD_COMPUTE_HH_
# define THREAD_COMPUTE_HH_

# include <thread>
# include <condition_variable>
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

    std::condition_variable& get_memory_cv()
    {
      return memory_cv_;
    }

    void request_refresh()
    {
      pipeline_->request_refresh();
    }

    void request_autofocus()
    {
      pipeline_->request_autofocus();
    }

    void request_autocontrast()
    {
      pipeline_->request_autocontrast();
    }
  private:
    void thread_proc();

  private:
    ComputeDescriptor& compute_desc_;
    Queue& input_;
    Queue& output_;
    Pipeline* pipeline_;

    bool compute_on_;
    std::condition_variable memory_cv_;
    std::thread thread_;
  };
}

#endif /* !THREAD_COMPUTE_HH_ */
