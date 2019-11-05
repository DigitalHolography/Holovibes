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
 * Thread encapsulation for obtaining images from a camera. */
#pragma once

#include <thread>

# include "ithread_input.hh"

/* Forward declarations. */
namespace holovibes
{
  class Queue;
}
namespace camera
{
  class ICamera;
}

namespace holovibes
{
  /*! \brief Thread encapsulation for obtaining images from a camera. */
  class ThreadCapture : public IThreadInput
  {
  public:
    /*! \brief Set a capture thread from a given camera and a destination queue.
     * \param The camera must be initialized
     * \param Destination queue */
    ThreadCapture(camera::ICamera& camera, Queue& input, SquareInputMode mode);

    ~ThreadCapture();

    const camera::FrameDescriptor& get_input_frame_descriptor() const override;
    const camera::FrameDescriptor& get_queue_frame_descriptor() const override;
  private:
    /*! While the thread is running, the get_frame() function (see ICamera
     * interface) is called with the current camera. Images sent are enqueued. */
    void thread_proc();

  private:
    camera::ICamera& camera_;

    Queue& queue_;

    std::thread thread_;
  };
}