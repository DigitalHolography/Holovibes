#pragma once

# include <thread>
# include <condition_variable>
# include <memory>

# include "queue.hh"
# include "compute_descriptor.hh"
# include "pipeline.hh"

namespace holovibes
{
  /*! \brief Thread managing pipeline
   *
   * While is running execute the pipeline
   */
  class ThreadCompute
  {
  public:
    /*! \brief Constructor
     *
     * params are gived to pipeline
     */
    ThreadCompute(
      ComputeDescriptor& desc,
      Queue& input,
      Queue& output,
      const bool is_float_output_enabled,
      const std::string float_output_file_src,
      const unsigned int float_output_nb_frame);

    ~ThreadCompute();

    /*! \return the running pipeline */
    std::shared_ptr<Pipeline> get_pipeline()
    {
      return pipeline_;
    }

    /*! \return condition_variable */
    std::condition_variable& get_memory_cv()
    {
      return memory_cv_;
    }

    /*! request pipeline refresh */
    void request_refresh()
    {
      pipeline_->request_refresh();
    }

    /*! request pipeline autofocus */
    void request_autofocus()
    {
      pipeline_->request_autofocus();
    }

    /*! request pipeline autocontrast */
    void request_autocontrast()
    {
      pipeline_->request_autocontrast();
    }

  private:
    /*! Execute pipeline while is running */
    void thread_proc(std::string float_output_file_src,
      const unsigned int float_output_nb_frame);

  private:
    ComputeDescriptor& compute_desc_;

    Queue& input_;
    Queue& output_;

    std::shared_ptr<Pipeline> pipeline_;

    /*! \brief Stored for the pipeline constructor*/
    bool is_float_output_enabled_;

    /*! \brief Is notify when pipeline is ready */
    std::condition_variable memory_cv_;
    std::thread thread_;
  };
}