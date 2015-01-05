#include "camera_ixon.hh"
#include <camera_exception.hh>

#include "atmcd32d.h"
#include <iostream>

namespace camera
{
  ICamera* new_camera_device()
  {
    return new CameraIxon();
  }

  CameraIxon::CameraIxon()
    : Camera("ixon.ini")
  {
    name_ = "ixon";
    long nb_cam;
    load_default_params();
    GetAvailableCameras(&nb_cam);
    if (nb_cam < 1)
      throw CameraException(CameraException::NOT_CONNECTED);

    if (ini_file_is_open())
      load_ini_params();
  }

  CameraIxon::~CameraIxon()
  {
  }

  void CameraIxon::init_camera()
  {
    GetCameraHandle(0, &device_handle);
    if (SetCurrentCamera(device_handle) == DRV_P1INVALID)
      throw CameraException(CameraException::NOT_INITIALIZED);
    char aBuffer[256];
    GetCurrentDirectory(256, aBuffer);
    if (Initialize(aBuffer) != DRV_SUCCESS)
      throw CameraException(CameraException::NOT_INITIALIZED);
    int x, y;
    GetDetector(&x, &y);
#if _DEBUG
    std::cout << x << "    " << y << std::endl;
#endif
    image_ = new unsigned short[desc_.frame_res()];
    bind_params();
   // r_x = desc_.width;
   // r_y = desc_.height;
    output_image_ = new unsigned short[desc_.frame_res()];
  }

  void CameraIxon::start_acquisition()
  {
    unsigned int error;
    error = StartAcquisition();
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
  }

  void CameraIxon::stop_acquisition()
  {
    AbortAcquisition();
  }

  void CameraIxon::shutdown_camera()
  {
    ShutDown();
    delete[] image_;
    delete[] output_image_;
  }

  void* CameraIxon::get_frame()
  {
    unsigned int error;
    long first;
    long last;
    GetNumberNewImages(&first, &last);
    std::cout << "first: " << first << " last: " << last << std::endl;
    if (trigger_mode_ == 10)
    {
      error = SendSoftwareTrigger();
      if (error != DRV_SUCCESS)
        throw CameraException(CameraException::CANT_GET_FRAME);
    }

    WaitForAcquisition();
    error = GetNewData16(image_, r_x * r_y);

    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_GET_FRAME);

    for (int y = 0; y < r_y; y++)
      memcpy(output_image_ + y * desc_.width, image_ + r_y * y, r_y * sizeof(unsigned short));
    return output_image_;
  }

  void CameraIxon::load_default_params()
  {
    desc_.width = 1024;
    desc_.height = 1024;
    r_x = 1002;
    r_y = 1002;
    desc_.depth = 2;
    desc_.pixel_size = 8.0f;
    desc_.endianness = LITTLE_ENDIAN;
    exposure_time_ = 0.1;
    trigger_mode_ = 10;
    shutter_close_ = 0;
    shutter_open_ = 0;
    ttl_ = 1;
    shutter_mode_ = 5;
    acquisiton_mode_ = 5;
    read_mode_ = 4;
    gain_mode_ = 0;
    kinetic_time_ = 0;
    horizontal_shift_speed_ = 0;
    vertical_shift_speed_ = 0;
  }

  void CameraIxon::load_ini_params()
  {
    /* Use the default value in case of fail. */
    const boost::property_tree::ptree& pt = get_ini_pt();

    r_x = pt.get<unsigned short>("ixon.sensor_width", desc_.width);
    r_y = pt.get<unsigned short>("ixon.sensor_height", desc_.height);

    exposure_time_ = pt.get<float>("ixon.exposure_time", exposure_time_);
    trigger_mode_ = pt.get<int>("ixon.trigger_mode", trigger_mode_);
    shutter_close_ = pt.get<float>("ixon.shutter_close", shutter_close_);
    shutter_open_ = pt.get<float>("ixon.shutter_open", shutter_close_);
    ttl_ = pt.get<int>("ixon.ttl", ttl_);
    shutter_mode_ = pt.get<int>("ixon.shutter_mode", shutter_mode_);
    acquisiton_mode_ = pt.get<int>("ixon.acquistion_mode", acquisiton_mode_);
    read_mode_ = pt.get<int>("ixon.read_mode", read_mode_);
    kinetic_time_ = pt.get<float>("ixon.kinetic_cycle_time", kinetic_time_);
    gain_mode_ = pt.get<int>("ixon.gain_mode", gain_mode_);
    horizontal_shift_speed_ = pt.get<float>("ixon.horizontal_shift_speed", horizontal_shift_speed_);
    vertical_shift_speed_ = pt.get<float>("ixon.vertical_shift_speed", vertical_shift_speed_);
  }
  void CameraIxon::bind_params()
  {
    unsigned int error;
    error = SetAcquisitionMode(acquisiton_mode_); // RUN TILL ABORT
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    error = SetReadMode(read_mode_);
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    error = SetExposureTime(exposure_time_);
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    error = SetShutter(ttl_, shutter_mode_, shutter_open_, shutter_close_);
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    error = SetTriggerMode(trigger_mode_);
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    error = SetImage(1, 1, 1, r_x, 1, r_y);
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    error = SetPreAmpGain(gain_mode_);
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    error = SetKineticCycleTime(kinetic_time_);
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    error = SetHSSpeed(0, horizontal_shift_speed_);
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    error = SetVSSpeed(vertical_shift_speed_);
    if (error != DRV_SUCCESS)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
  }
}