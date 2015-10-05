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
    roi_x_ = 0;
    roi_y_ = 0;
    roi_width_ = 2048;
    roi_height_ = 2048;

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
    /* Updating frame size, taking account downsampling. */
    desc_.width = desc_.width / downsampling_rate_;
    desc_.height = desc_.height / downsampling_rate_;

    std::string str;

    str = pt.get<std::string>("xiq.downsampling_type", "");
    if (str == "BINNING")
      downsampling_type_ = XI_BINNING;
    else if (str == "SKIPPING")
      downsampling_type_ = XI_SKIPPING;

    /* Making sure ROI settings are valid. */
    {
      int tmp_roi_x = pt.get<int>("xiq.roi_x", roi_x_);
      int tmp_roi_y = pt.get<int>("xiq.roi_y", roi_y_);
      int tmp_roi_width = pt.get<int>("xiq.roi_width", roi_width_);
      int tmp_roi_height = pt.get<int>("xiq.roi_height", roi_height_);

      if (tmp_roi_x < desc_.width ||
        tmp_roi_y < desc_.height ||
        tmp_roi_width <= desc_.width ||
        tmp_roi_height <= desc_.height)
      {
        roi_x_ = tmp_roi_x;
        roi_y_ = tmp_roi_y;
        roi_width_ = tmp_roi_width;
        roi_height_ = tmp_roi_height;

        // Don't forget to update the frame descriptor!
        desc_.width = roi_width_;
        desc_.height = roi_height_;
      }
      else
        std::cerr << "[CAMERA] Invalid ROI settings, ignoring ROI." << std::endl;
    }
    
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
    status |= xiSetParamInt(device_, XI_PRM_AEAG_ROI_OFFSET_X, roi_x_);
    status |= xiSetParamInt(device_, XI_PRM_AEAG_ROI_OFFSET_Y, roi_y_);
    status |= xiSetParamInt(device_, XI_PRM_AEAG_ROI_WIDTH, roi_width_);
    status |= xiSetParamInt(device_, XI_PRM_AEAG_ROI_HEIGHT, roi_height_);

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
