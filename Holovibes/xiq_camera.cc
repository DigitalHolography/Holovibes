#include "stdafx.h"
#include "xiq_camera.hh"

#include <cassert>

namespace camera
{
  XiqCamera::XiqCamera()
    : Camera("Xiq")
    , device_(nullptr)
    , status_(XI_OK)
  {
    frame_.size = sizeof (XI_IMG);
    frame_.bp = nullptr;
    frame_.bp_size = 0;
  }

  bool XiqCamera::init_camera()
  {
    status_ = xiOpenDevice(0, &device_);
    assert(xiSetParamInt(device_, XI_PRM_DOWNSAMPLING, 1L) == XI_OK);
    xiSetParamInt(device_, XI_PRM_DOWNSAMPLING_TYPE, 1);
    xiSetParamInt(device_, XI_PRM_IMAGE_DATA_FORMAT, XI_RAW8);
    xiSetParamInt(device_, XI_PRM_BUFFER_POLICY, XI_BP_SAFE);
    xiSetParamInt(device_, XI_PRM_EXPOSURE, (int)((double) 1.0e6 * 0.005));

    return status_ == XI_OK;
  }

  void XiqCamera::start_acquisition()
  {
    status_ = xiStartAcquisition(device_);
  }

  void XiqCamera::stop_acquisition()
  {
    status_ = xiStopAcquisition(device_);
  }

  void XiqCamera::shutdown_camera()
  {
    status_ = xiCloseDevice(device_);
  }

  void* XiqCamera::get_frame()
  {
    status_ = xiGetImage(device_, 1000, &frame_);
    printf("[FRAME][NEW] %dx%d - %u\n",
      frame_.width,
      frame_.height,
      frame_.nframe);

    return frame_.bp;
  }
}