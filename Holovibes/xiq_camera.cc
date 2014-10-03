#include "stdafx.h"
#include "xiq_camera.hh"
#include "exception_camera.hh"

#include <cassert>

namespace camera
{
  XiqCamera::XiqCamera()
    : Camera()
    , device_(nullptr)
  {
    frame_.size = sizeof(XI_IMG);
    frame_.bp = nullptr;
    frame_.bp_size = 0;
  }

  bool XiqCamera::init_camera()
  {
    if (xiOpenDevice(0, &device_) != XI_OK)
      throw ExceptionCamera(name_, ExceptionCamera::NOT_INITIALIZED);

    load_param();

    // TODO: Unused return.
    return true;
  }

  void XiqCamera::load_param()
  {
    XI_RETURN status = XI_OK;

    status = xiSetParamInt(device_, XI_PRM_DOWNSAMPLING, 1L);
    status = xiSetParamInt(device_, XI_PRM_DOWNSAMPLING_TYPE, 1);
    status = xiSetParamInt(device_, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8);
    status = xiSetParamInt(device_, XI_PRM_BUFFER_POLICY, XI_BP_SAFE);
    status = xiSetParamInt(device_, XI_PRM_EXPOSURE, (int)((double) 1.0e6 * 0.005));

    desc_.height = 2048;
    desc_.width = 2048;

    if (status != XI_OK)
      throw ExceptionCamera(name_, ExceptionCamera::CANT_SET_CONFIG);
  }

  void XiqCamera::start_acquisition()
  {
    if (xiStartAcquisition(device_) != XI_OK)
      throw ExceptionCamera(name_, ExceptionCamera::CANT_START_ACQUISITION);
  }

  void XiqCamera::stop_acquisition()
  {
    if (xiStopAcquisition(device_) != XI_OK)
      throw ExceptionCamera(name_, ExceptionCamera::CANT_STOP_ACQUISITION);
  }

  void XiqCamera::shutdown_camera()
  {
    if (xiCloseDevice(device_) != XI_OK)
      throw ExceptionCamera(name_, ExceptionCamera::CANT_SHUTDOWN);
  }

  void* XiqCamera::get_frame()
  {
    if (xiGetImage(device_, FRAME_TIMEOUT, &frame_) != XI_OK)
      throw ExceptionCamera(name_, ExceptionCamera::CANT_GET_FRAME);

    printf("[FRAME][NEW] %dx%d - %u\n",
      frame_.width,
      frame_.height,
      frame_.nframe);

    return frame_.bp;
  }
}