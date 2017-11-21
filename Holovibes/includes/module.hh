/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
 *
 * Regroups one or several predefined tasks,  which shall work
 * sequentially on a single data buffer. */
#pragma once

# include "pipeline_utils.hh"

namespace holovibes
{
  /*! Regroups one or several predefined tasks,  which shall work
   * sequentially on a single data buffer.
   *
   * A Module uses its own thread to work on these tasks.
   * A Module needs to be handled by a upper level manager,
   * without which it will not synchronize with other Modules. */
  class Module
  {
  public:
    /*! Initialize a module with no tasks, and the address
     * to a boolean value managing its activity.
     *
     * The newly created Module automatically creates a new CUDA stream
     * to use for itself. */
    Module();

    //!< Join the thread before exiting.
    ~Module();

    //!< Add an extra task after others tasks, to carry at each iteration.
    void  push_back_worker(FnType worker);
    //!< Add an extra task before others tasks, to carry at each iteration.
    void  push_front_worker(FnType worker);

    //!< The function used by the created thread.
    void  thread_proc();

  public:
    //!< Boolean managing activity/waiting.
    bool          task_done_;
    //!< Each Modules need stream given to their worker
    cudaStream_t  stream_;
  private:
    //!< Set to true by the Module to stop and join its thread, when asked to stop.
    bool        stop_requested_;
    //!< All tasks are carried out sequentially, in growing index order.
    FnDeque     workers_;
    std::thread thread_;
  };
}