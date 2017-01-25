/*! \file
 *
 * Thread encapsulation for managing all the processing performed
 * in hologram rendering mode. */
#pragma once

# include <thread>
# include <condition_variable>
# include <memory>

# include "icompute.hh"

/* Forward declarations. */
namespace holovibes
{
  class ComputeDescriptor;
  class Queue;
}

namespace holovibes
{
  /*! \brief Thread encapsulation for managing all the processing performed
   * in hologram rendering mode.
   *
   * Executes in a loop an instance of the ICompute class, and handles its
   * stopping and destruction.
   */
  class ThreadCompute
  {
  public:

    /*! All types of ICompute-based classes are associated with a flag. */
    enum PipeType
    {
      PIPE,
      PIPELINE,
    };

    /*! \brief Build an ICompute instance between two queues.
     *
     * \param desc The Compute Descriptor which will be used and modified
     * by the ICompute instance.
     */
    ThreadCompute(
      ComputeDescriptor& desc,
      Queue& input,
      Queue& output,
      const PipeType pipetype);

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
    void thread_proc();

  private:
    /*! ComputeDescriptor used by the ICompute object.  */
    ComputeDescriptor& compute_desc_;

    Queue& input_;
    Queue& output_;
    /*! The current type of ICompute object used. */
    const PipeType pipetype_;

    std::shared_ptr<ICompute> pipe_;

    /*! \brief Is notified when the ICompute object is ready */
    std::condition_variable memory_cv_;
    std::thread thread_;
  };
}