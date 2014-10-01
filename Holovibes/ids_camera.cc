#include "stdafx.h"
#include "ids_camera.hh"

namespace camera
{
  bool IDSCamera::init_camera()
  {
    int cameras_nb = 0;
    int result = is_GetNumberOfCameras(&cameras_nb);

    if (result == IS_SUCCESS && cameras_nb > 0)
    {
      // Assuming there's only one camera connected.
      cam_ = 0;
      result = is_InitCamera(&cam_, nullptr);

      if (result == IS_SUCCESS)
      {
        // Memory allocation
        is_AllocImageMem(cam_,
          desc_.width,
          desc_.height,
          desc_.bit_depth,
          &frame_,
          &frame_mem_pid_);

        // Setting color mode at 8 bits (grey scale)
        is_SetColorMode(cam_, IS_CM_SENSOR_RAW8);
      }
      else
        throw new CameraException(name_, CameraException::camera_error::NOT_INITIALIZED);
    }
    else
      throw new CameraException(name_, CameraException::camera_error::NOT_CONNECTED);

    return result == IS_SUCCESS;
  }

  void IDSCamera::start_acquisition()
  {
    stop_acquisition();
    int result = is_SetImageMem(cam_, frame_, frame_mem_pid_);

    if (result != IS_SUCCESS)
      throw new CameraException(name_, CameraException::camera_error::MEMORY_PROBLEM);
  }

  void IDSCamera::stop_acquisition()
  {
  }

  void IDSCamera::shutdown_camera()
  {
    int result = IS_SUCCESS;
    result = is_FreeImageMem(cam_, frame_, frame_mem_pid_);
    result = is_ExitCamera(cam_);

    if (result != IS_SUCCESS)
      throw new CameraException(name_, CameraException::camera_error::CANT_SHUTDOWN);
  }

  void* IDSCamera::get_frame()
  {
    if (is_FreezeVideo(cam_, IS_WAIT) != IS_SUCCESS)
      throw new CameraException(name_, CameraException::camera_error::CANT_GET_FRAME);

    return frame_;
  }
}