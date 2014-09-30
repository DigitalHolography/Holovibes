#include "stdafx.h"
#include "xiq_camera.hh"

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
    XI_RETURN status = XI_OK;
    status = xiOpenDevice(0, &device_);

    load_param();

    return status == XI_OK;
  }

  void XiqCamera::load_param()
  {
    XI_RETURN status = XI_OK;

    status = xiSetParamInt(device_, XI_PRM_DOWNSAMPLING, 1L);
    status = xiSetParamInt(device_, XI_PRM_DOWNSAMPLING_TYPE, 1);
    status = xiSetParamInt(device_, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8);
    status = xiSetParamInt(device_, XI_PRM_BUFFER_POLICY, XI_BP_SAFE);
    status = xiSetParamInt(device_, XI_PRM_EXPOSURE, (int)((double) 1.0e6 * 0.005));
  }

  void XiqCamera::start_acquisition()
  {
    xiStartAcquisition(device_);
  }

  void XiqCamera::stop_acquisition()
  {
    xiStopAcquisition(device_);
  }

  void XiqCamera::shutdown_camera()
  {
    xiCloseDevice(device_);
  }

  void* XiqCamera::get_frame()
  {
    xiGetImage(device_, FRAME_TIMEOUT, &frame_);
    printf("[FRAME][NEW] %dx%d - %u\n",
      frame_.width,
      frame_.height,
      frame_.nframe);

    return frame_.bp;
  }
}