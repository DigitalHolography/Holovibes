#ifndef THREAD_COMPUTE_HH_
# define THREAD_COMPUTE_HH_

# include <thread>
# include <condition_variable>
# include <memory>

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
      Queue& output,
      bool is_float_output_enabled,
      std::string float_output_file_src,
      unsigned int float_output_nb_frame);
    ~ThreadCompute();

    std::shared_ptr<Pipeline> get_pipeline()
    {
      return pipeline_;
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
    void thread_proc(std::string float_output_file_src, unsigned int float_output_nb_frame);

  private:
    ComputeDescriptor& compute_desc_;
    Queue& input_;
    Queue& output_;
    std::shared_ptr<Pipeline> pipeline_;

    bool is_float_output_enabled_;
    std::condition_variable memory_cv_;
    std::thread thread_;
  };
}

#endif /* !THREAD_COMPUTE_HH_ */