#include "pike_camera.hh"

#define MAXNAMELENGTH 256

namespace camera
{

  bool PikeCamera::init_camera()
  {
    if (cam_.StartDevice() != 0)
    {
      return false;
    }

    name_ = get_name_from_device();

    return true;
  }

  void PikeCamera::start_acquisition()
  {
  }

  void PikeCamera::stop_acquisition()
  {

  }

  std::string PikeCamera::get_name_from_device()
  {
    char* ccam_name = new char[MAXNAMELENGTH];

    if (cam_.GetDeviceName(ccam_name, MAXNAMELENGTH) != 0)
      return "";

    std::string cam_name(ccam_name);

    return cam_name;
  }

  void PikeCamera::shutdown_camera()
  {

  }
}