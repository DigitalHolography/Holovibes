#include "stdafx.h"
#include "xiq_camera.hh"
#include "exception_camera.hh"

#include <cassert>

namespace camera
{
  XiqCamera::XiqCamera()
    : Camera("xiq.ini")
    , device_(nullptr)
  {
    load_default_params();
    if (ini_file_is_open())
      load_ini_params();

    frame_.size = sizeof(XI_IMG);
    frame_.bp = nullptr;
    frame_.bp_size = 0;
  }

  void XiqCamera::init_camera()
  {
    if (xiOpenDevice(0, &device_) != XI_OK)
      throw ExceptionCamera(name_, ExceptionCamera::NOT_INITIALIZED);

    /* Configure the camera API with given parameters. */
    bind_params();
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

  void XiqCamera::load_default_params()
  {
    name_ = "Xiq";
    exposure_time_ = 0.005;
    /* Custom parameters. */
    downsampling_rate_ = 1;
    downsampling_type_ = XI_SKIPPING;
    img_format_ = XI_RAW8;
    buffer_policy_ = XI_BP_SAFE;
 }

  void XiqCamera::load_ini_params()
  {
    const boost::property_tree::ptree& pt = get_ini_pt();

    /* Use the default value in case of fail. */
    exposure_time_ = pt.get<double>("xiq.exposure_time", exposure_time_);
    downsampling_rate_ = pt.get<int>("xiq.downsampling_rate", downsampling_rate_);

    std::string str;

    str = pt.get<std::string>("xiq.downsampling_type", "");
    if (str.compare("BINNING"))
      downsampling_type_ = XI_BINNING;
    else if (str.compare("SKIPPING"))
      downsampling_type_ = XI_SKIPPING;

    str = pt.get<std::string>("xiq.format", "");
    if (str.compare("XI_MONO8"))
      img_format_ = XI_MONO8;
    else if (str.compare("XI_MONO16"))
      img_format_ = XI_MONO16;
    else if (str.compare("XI_RAW8"))
      img_format_ = XI_RAW8;
    else if (str.compare("XI_RAW16"))
      img_format_ = XI_RAW16;

    std::cout << downsampling_type_;
    std::cout << exposure_time_ << std::endl;
  }
  
  void XiqCamera::bind_params()
  {
    XI_RETURN status = XI_OK;

    char name[32];

    status = xiGetParamString(device_, XI_PRM_DEVICE_NAME, &name, 32);
    status = xiSetParamInt(device_, XI_PRM_DOWNSAMPLING, downsampling_rate_);
    status = xiSetParamInt(device_, XI_PRM_DOWNSAMPLING_TYPE, downsampling_type_);
    status = xiSetParamInt(device_, XI_PRM_IMAGE_DATA_FORMAT, img_format_);
    status = xiSetParamInt(device_, XI_PRM_BUFFER_POLICY, buffer_policy_);
    status = xiSetParamInt(device_, XI_PRM_EXPOSURE, (int)((double) 1.0e6 * exposure_time_));

    if (status != XI_OK)
      throw ExceptionCamera(name_, ExceptionCamera::CANT_SET_CONFIG);

    name_ = std::string(name);
  }
}
