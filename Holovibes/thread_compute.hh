#pragma once

# include <thread>
# include <condition_variable>
# include <memory>

# include "queue.hh"
# include "compute_descriptor.hh"
# include "icompute.hh"

namespace holovibes
{
  /*! \brief Thread managing pipe
   *
   * While is running execute the pipe
   */
  class ThreadCompute
  {
  public:

    enum PipeType
    {
      PIPE,
      PIPELINE,
    };

    /*! \brief Constructor
     *
     * params are gived to pipe
     */
    ThreadCompute(
      ComputeDescriptor& desc,
      Queue& input,
      Queue& output,
      const PipeType pipetype,
      const bool is_float_output_enabled,
      const std::string float_output_file_src,
      const unsigned int float_output_nb_frame);

    ~ThreadCompute();

    /*! \return the running pipe */
    std::shared_ptr<ICompute> get_pipe()
    {
      return pipe_;
    }

    /*! \return condition_variable */
    std::condition_variable& get_memory_cv()
    {
      return memory_cv_;
    }

    /*! request pipe refresh */
    void request_refresh()
    {
      pipe_->request_refresh();
    }

    /*! request pipe autofocus */
    void request_autofocus()
    {
      pipe_->request_autofocus();
    }

    /*! request pipe autocontrast */
    void request_autocontrast()
    {
      pipe_->request_autocontrast();
    }

  private:
    /*! Execute pipe while is running */
    void thread_proc(std::string float_output_file_src,
      const unsigned int float_output_nb_frame);

  private:
    ComputeDescriptor& compute_desc_;

    Queue& input_;
    Queue& output_;
    const PipeType pipetype_;

    std::shared_ptr<ICompute> pipe_;

    /*! \brief Stored for the pipe constructor*/
    bool is_float_output_enabled_;

    /*! \brief Is notify when pipe is ready */
    std::condition_variable memory_cv_;
    std::thread thread_;
  };
}