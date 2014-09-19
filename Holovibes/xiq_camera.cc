#include "xiq_camera.hh"

namespace camera
{
  bool XiqCamera::init_camera()
  {
    status_ = xiOpenDevice(0, &device_);
    return status_ == XI_OK;
  }
}