#include "stdafx.h"
#include "xiq_camera.hh"

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
    status_ = xiGetImage(device_, 10000, &frame_);
    std::cout << "new frame " << frame_.width << "x"
      << frame_.height << frame_.nframe << std::endl;
    return frame_.bp;
  }
}