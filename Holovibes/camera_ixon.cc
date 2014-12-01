#include "camera_ixon.hh"
#include "camera_exception.hh"

namespace camera
{
  CameraIxon::CameraIxon()
    : Camera("Ixon.ini")
  {
    name_ = "ixon";
    long nb_cam;
    load_default_params();
    GetAvailableCameras(&nb_cam);
    if (nb_cam < 1)
      throw CameraException(name_, CameraException::NOT_CONNECTED);

    std::cout << nb_cam << std::endl;//remove me

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
    std::cout << x << "    " << y << std::endl;
    image_ = (unsigned short*)malloc(r_x * r_y * sizeof(unsigned short));
   // r_x = desc_.width;
   // r_y = desc_.height;
    output_image_ = (unsigned short *)malloc(desc_.frame_size());
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
    error = SetImage(1, 1, 1, /*desc_.width*/ r_x, 1, r_y /*desc_.height*/);
    if (error != DRV_SUCCESS)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
    //SetCoolerMode(0);
    StartAcquisition();
  }

  void CameraIxon::stop_acquisition()
  {
  }

  void CameraIxon::shutdown_camera()
  {
    ShutDown();
    free(image_);
    free(output_image_);
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
    error = GetNewData16(image_, r_x * r_y);

    if (error != DRV_SUCCESS && error != DRV_NO_NEW_DATA)
      throw CameraException(name_, CameraException::CANT_GET_FRAME);
    else
    {
      for (int y = 0; y < r_y; y++)
        memcpy(output_image_ + y * desc_.width , image_ + r_y * y , r_y * sizeof(unsigned short));
      //return ((void*)image_);
      return ((void *)output_image_);
    }
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
  }
  void CameraIxon::bind_params()
  {
  }
}