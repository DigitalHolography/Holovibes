#include "camera_xiq.hh"
#include <camera_exception.hh>

namespace camera
{
  ICamera* new_camera_device()
  {
    return new CameraXiq();
  }

  CameraXiq::CameraXiq()
    : Camera("xiq.ini")
    , device_(nullptr)
  {
    name_ = "xiq";
    load_default_params();
    if (ini_file_is_open())
      load_ini_params();

    frame_.size = sizeof(XI_IMG);
    frame_.bp = nullptr;
    frame_.bp_size = 0;
  }

  void CameraXiq::init_camera()
  {
    if (xiOpenDevice(0, &device_) != XI_OK)
      throw CameraException(CameraException::NOT_INITIALIZED);

    /* Configure the camera API with given parameters. */
    bind_params();
  }

  void CameraXiq::start_acquisition()
  {
    if (xiStartAcquisition(device_) != XI_OK)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
  }

  void CameraXiq::stop_acquisition()
  {
    if (xiStopAcquisition(device_) != XI_OK)
      throw CameraException(CameraException::CANT_STOP_ACQUISITION);
  }

  void CameraXiq::shutdown_camera()
  {
    if (xiCloseDevice(device_) != XI_OK)
      throw CameraException(CameraException::CANT_SHUTDOWN);
  }

  void* CameraXiq::get_frame()
  {
    if (xiGetImage(device_, FRAME_TIMEOUT, &frame_) != XI_OK)
      throw CameraException(CameraException::CANT_GET_FRAME);

#if 0
    printf("[FRAME][NEW] %dx%d - %u\n",
      frame_.width,
      frame_.height,
      frame_.nframe);
#endif

    return frame_.bp;
  }

  void CameraXiq::load_default_params()
  {
    exposure_time_ = 0.005f;
    /* Custom parameters. */
    gain_ = 0.f;
    downsampling_rate_ = 1;
    downsampling_type_ = XI_BINNING;
    img_format_ = XI_RAW8;
    buffer_policy_ = XI_BP_SAFE;

    /* Fill the frame descriptor. */
    desc_.width = 2048;
    desc_.height = 2048;
    desc_.pixel_size = 5.5f;
    desc_.depth = 1;
    desc_.endianness = BIG_ENDIAN;
  }

  void CameraXiq::load_ini_params()
  {
    const boost::property_tree::ptree& pt = get_ini_pt();

    /* Use the default value in case of fail. */
    exposure_time_ = pt.get<float>("xiq.exposure_time", exposure_time_);
    gain_ = pt.get<float>("xiq.gain", gain_);
    downsampling_rate_ = pt.get<int>("xiq.downsampling_rate", downsampling_rate_);

    std::string str;

    str = pt.get<std::string>("xiq.downsampling_type", "");
    if (str == "BINNING")
      downsampling_type_ = XI_BINNING;
    else if (str == "SKIPPING")
      downsampling_type_ = XI_SKIPPING;

    str = pt.get<std::string>("xiq.format", "");
    if (str == "MONO8")
      img_format_ = XI_MONO8;
    else if (str == "MONO16")
      img_format_ = XI_MONO16;
    else if (str == "RAW8")
      img_format_ = XI_RAW8;
    else if (str == "RAW16")
      img_format_ = XI_RAW16;

    trigger_src_ = (XI_TRG_SOURCE)pt.get<unsigned long>("xiq.trigger_src", XI_TRG_OFF);
  }

  void CameraXiq::bind_params()
  {
    XI_RETURN status = XI_OK;

    const unsigned int name_buffer_size = 32;
    char name[name_buffer_size];

    status |= xiGetParamString(device_, XI_PRM_DEVICE_NAME, &name, name_buffer_size);
    status |= xiSetParamInt(device_, XI_PRM_DOWNSAMPLING, downsampling_rate_);
    status |= xiSetParamInt(device_, XI_PRM_DOWNSAMPLING_TYPE, downsampling_type_);
    status |= xiSetParamInt(device_, XI_PRM_IMAGE_DATA_FORMAT, img_format_);
    status |= xiSetParamInt(device_, XI_PRM_BUFFER_POLICY, buffer_policy_);
    status |= xiSetParamFloat(device_, XI_PRM_EXPOSURE, 1.0e6f * exposure_time_);
    status |= xiSetParamFloat(device_, XI_PRM_GAIN, gain_);
    status |= xiSetParamInt(device_, XI_PRM_TRG_SOURCE, trigger_src_);

    if (status != XI_OK)
      throw CameraException(CameraException::CANT_SET_CONFIG);

    /* Update the frame descriptor. */
    if (img_format_ == XI_RAW16 || img_format_ == XI_MONO16)
      desc_.depth = 2;

    name_ = std::string(name);
  }
}
