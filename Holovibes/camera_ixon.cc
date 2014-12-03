#include "camera_ixon.hh"
#include "camera_exception.hh"

#include "atmcd32d.h"

namespace camera
{
  CameraIxon::CameraIxon()
    : Camera("ixon.ini")
  {
    name_ = "ixon";
    long nb_cam;
    load_default_params();
    GetAvailableCameras(&nb_cam);
    if (nb_cam < 1)
      throw CameraException(name_, CameraException::NOT_CONNECTED);

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
      throw CameraException(name_, CameraException::NOT_INITIALIZED);
    char aBuffer[256];
    GetCurrentDirectory(256, aBuffer);
    if (Initialize(aBuffer) != DRV_SUCCESS)
      throw CameraException(name_, CameraException::NOT_INITIALIZED);
    int x, y;
    GetDetector(&x, &y);
#if _DEBUG
    std::cout << x << "    " << y << std::endl;
#endif
    image_ = new unsigned short[desc_.frame_res()];
    /* CALL BIND PARAMS ! */
    bind_params();
  }

  void CameraIxon::start_acquisition()
  {
    unsigned int error;
    error = SetAcquisitionMode(acquisiton_mode_); // RUN TILL ABORT
    if (error != DRV_SUCCESS)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
    error = SetReadMode(read_mode_);
    if (error != DRV_SUCCESS)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
    error = SetExposureTime(exposure_time_);
    if (error != DRV_SUCCESS)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
    error = SetShutter(ttl_, shutter_mode_, shutter_open_, shutter_close_);
    if (error != DRV_SUCCESS)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
    error = SetTriggerMode(trigger_mode_);
    if (error != DRV_SUCCESS)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
    error = SetImage(1, 1, 1, desc_.width, 1, desc_.height);
    if (error != DRV_SUCCESS)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);

#if _DEBUG
    /* FIXME: Enable this. */
    //SetCoolerMode(0);
#endif

    /* FIXME: Do error checking on start acquisition. */
    StartAcquisition();
  }

  void CameraIxon::stop_acquisition()
  {
    /* FIXME */
    //AbortAcquisition();
  }

  void CameraIxon::shutdown_camera()
  {
    ShutDown();
    delete[] image_;
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
        throw CameraException(name_, CameraException::CANT_GET_FRAME);
    }

    WaitForAcquisition();
    error = GetNewData16(image_, desc_.width * desc_.height);

    if (error != DRV_SUCCESS)
      throw CameraException(name_, CameraException::CANT_GET_FRAME);

    return image_;
  }

  void CameraIxon::load_default_params()
  {
    desc_.width = 1002;
    desc_.height = 1002;
    desc_.depth = 2;
    desc_.pixel_size = 7.4f;
    desc_.endianness = LITTLE_ENDIAN;
    exposure_time_ = 0.1;
    trigger_mode_ = 10; //0
    shutter_close_ = 0;
    shutter_open_ = 0;
    ttl_ = 1;
    shutter_mode_ = 5; //0
    acquisiton_mode_ = 5;
    read_mode_ = 4;
  }

  void CameraIxon::load_ini_params()
  {
    /* Use the default value in case of fail. */
    const boost::property_tree::ptree& pt = get_ini_pt();

    /* Set width/height in INI file is bad (what happens if the user set weird values ?). */
    desc_.width = pt.get<unsigned short>("ixon.sensor_width", desc_.width);
    desc_.height = pt.get<unsigned short>("ixon.sensor_height", desc_.height);

    exposure_time_ = pt.get<float>("ixon.exposure_time", exposure_time_);
    trigger_mode_ = pt.get<int>("ixon.trigger_mode", trigger_mode_);
    shutter_close_ = pt.get<float>("ixon.shutter_close", shutter_close_);
    shutter_open_ = pt.get<float>("ixon.shutter_open", shutter_close_);
    ttl_ = pt.get<int>("ixon.ttl", ttl_);
    shutter_mode_ = pt.get<int>("ixon.shutter_mode", shutter_mode_);
    acquisiton_mode_ = pt.get<int>("ixon.acquistion_mode", acquisiton_mode_);
    read_mode_ = pt.get<int>("ixon.read_mode", read_mode_);
  }
  void CameraIxon::bind_params()
  {
    /* FIXME: Where is this stuff ?
     * Start acquisition do too much things. */
  }
}