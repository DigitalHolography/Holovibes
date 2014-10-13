#include "stdafx.h"
#include "camera_ids.hh"
#include "camera_exception.hh"

namespace camera
{
  void CameraIds::init_camera()
  {
    int cameras_nb = 0;
    int result = is_GetNumberOfCameras(&cameras_nb);

    if (result == IS_SUCCESS && cameras_nb > 0)
    {
      // Assuming there's only one camera connected.
      cam_ = 0;
      result = is_InitCamera(&cam_, nullptr);

      if (result == IS_SUCCESS)
      {
        // Memory allocation
        is_AllocImageMem(cam_,
          desc_.width,
          desc_.height,
          desc_.bit_depth,
          &frame_,
          &frame_mem_pid_);
      }
      else
        throw new CameraException(name_, CameraException::camera_error::NOT_INITIALIZED);
    }
    else
      throw new CameraException(name_, CameraException::camera_error::NOT_CONNECTED);

    bind_params();
  }

  void CameraIds::start_acquisition()
  {
    stop_acquisition();

    if (is_SetImageMem(cam_, frame_, frame_mem_pid_) != IS_SUCCESS)
      throw new CameraException(name_, CameraException::camera_error::MEMORY_PROBLEM);
  }

  void CameraIds::stop_acquisition()
  {
  }

  void CameraIds::shutdown_camera()
  {
    if (is_FreeImageMem(cam_, frame_, frame_mem_pid_) != IS_SUCCESS)
      throw new CameraException(name_, CameraException::camera_error::MEMORY_PROBLEM);

    if (is_ExitCamera(cam_) != IS_SUCCESS)
      throw new CameraException(name_, CameraException::camera_error::CANT_SHUTDOWN);
  }

  void* CameraIds::get_frame()
  {
    if (is_FreezeVideo(cam_, IS_WAIT) != IS_SUCCESS)
      throw new CameraException(name_, CameraException::camera_error::CANT_GET_FRAME);

    return frame_;
  }

  void CameraIds::load_default_params()
  {
    desc_.width = 2048;
    desc_.height = 2048;
    desc_.endianness = BIG_ENDIAN;
    desc_.bit_depth = 8;

    exposure_time_ = 49.91f;
    gain_ = 0;
    subsampling_ = -1;
    binning_ = -1;
    color_mode_ = IS_CM_SENSOR_RAW8;

    frame_rate_ = 0;
  }

  void CameraIds::load_ini_params()
  {
    const boost::property_tree::ptree& pt = get_ini_pt();

    desc_.width = pt.get<int>("ids.sensor_width", 2048);
    desc_.height = pt.get<int>("ids.sensor_height", 2048);
    desc_.bit_depth;
    desc_.endianness;
    desc_.pixel_size;

    exposure_time_ = pt.get<float>("ids.exposure_time", exposure_time_);
    gain_ = pt.get<int>("ids.gain", gain_);
    format_gain();
    subsampling_ = get_subsampling_mode(pt.get<std::string>("ids.subsampling", ""));
    binning_ = get_binning_mode(pt.get<std::string>("ids.binning", ""));
    color_mode_ = get_color_mode(pt.get<std::string>("ids.image_format", ""));
    aoi_x = pt.get<int>("ids.aoi_startx", 0);
    aoi_y = pt.get<int>("ids.aoi_starty", 0);
    aoi_width_ = pt.get<int>("ids.aoi_width", 2048);
    aoi_height_ = pt.get<int>("ids.aoi_height", 2048);
    trigger_mode_ = get_trigger_mode(pt.get<std::string>("ids.trigger", ""));
  }

  void CameraIds::bind_params()
  {
    int status = IS_SUCCESS;

    // Exposure time
    // is_Exposure require a double as third argument.
    double exp = (double)exposure_time_;
    status = is_Exposure(cam_, IS_EXPOSURE_CMD_SET_EXPOSURE, &exp, sizeof(&exp));

    // Gain
    if (is_SetGainBoost(cam_, IS_GET_SUPPORTED_GAINBOOST) == IS_SET_GAINBOOST_ON)
    {
      status = is_SetGainBoost(cam_, IS_SET_GAINBOOST_ON);
      status = is_SetHardwareGain(cam_, gain_,
        IS_IGNORE_PARAMETER,
        IS_IGNORE_PARAMETER,
        IS_IGNORE_PARAMETER);
    }

    // Subsampling
    if (subsampling_ != -1)
      status = is_SetSubSampling(cam_, subsampling_);

    // Binning
    if (binning_ != -1)
      status = is_SetBinning(cam_, binning_);

    // Image format/Color mode
    status = is_SetColorMode(cam_, color_mode_);

    // Area Of Interest
    IS_RECT rectAOI;
    rectAOI.s32X = aoi_x;
    rectAOI.s32Y = aoi_y;
    rectAOI.s32Width = aoi_width_;
    rectAOI.s32Height = aoi_height_;

    status = is_AOI(cam_, IS_AOI_IMAGE_SET_AOI, (void*)&rectAOI, sizeof(rectAOI));

    // Trigger
    status = is_SetExternalTrigger(cam_, trigger_mode_);

    if (status != IS_SUCCESS)
      throw new CameraException(name_, CameraException::camera_error::CANT_SET_CONFIG);
  }

  int CameraIds::format_gain()
  {
    if (gain_ < 0 || gain_ > 100)
      return 0;
    return gain_;
  }

  int CameraIds::get_subsampling_mode(std::string ui)
  {
    if (ui == "2x2")
      return IS_SUBSAMPLING_2X_VERTICAL | IS_SUBSAMPLING_2X_HORIZONTAL;
    else if (ui == "3x3")
      return IS_SUBSAMPLING_3X_VERTICAL | IS_SUBSAMPLING_3X_HORIZONTAL;
    else if (ui == "4x4")
      return IS_SUBSAMPLING_4X_VERTICAL | IS_SUBSAMPLING_4X_HORIZONTAL;
    else if (ui == "5x5")
      return IS_SUBSAMPLING_5X_VERTICAL | IS_SUBSAMPLING_5X_HORIZONTAL;
    else if (ui == "6x6")
      return IS_SUBSAMPLING_6X_VERTICAL | IS_SUBSAMPLING_6X_HORIZONTAL;
    else if (ui == "8x8")
      return IS_SUBSAMPLING_8X_VERTICAL | IS_SUBSAMPLING_8X_HORIZONTAL;
    else if (ui == "16x16")
      return IS_SUBSAMPLING_16X_VERTICAL | IS_SUBSAMPLING_16X_HORIZONTAL;
    else
      return IS_SUBSAMPLING_DISABLE;
  }

  int CameraIds::get_binning_mode(std::string ui)
  {
    if (ui == "2x2")
      return IS_BINNING_2X_VERTICAL | IS_BINNING_2X_HORIZONTAL;
    else if (ui == "3x3")
      return IS_BINNING_3X_VERTICAL | IS_BINNING_3X_HORIZONTAL;
    else if (ui == "4x4")
      return IS_BINNING_4X_VERTICAL | IS_BINNING_4X_HORIZONTAL;
    else if (ui == "5x5")
      return IS_BINNING_5X_VERTICAL | IS_BINNING_5X_HORIZONTAL;
    else if (ui == "6x6")
      return IS_BINNING_6X_VERTICAL | IS_BINNING_6X_HORIZONTAL;
    else if (ui == "8x8")
      return IS_BINNING_8X_VERTICAL | IS_BINNING_8X_HORIZONTAL;
    else if (ui == "16x16")
      return IS_BINNING_16X_VERTICAL | IS_BINNING_16X_HORIZONTAL;
    else
      return IS_BINNING_DISABLE;
  }

  int CameraIds::get_color_mode(std::string ui)
  {
    if (ui == "RAW8")
      return IS_CM_SENSOR_RAW8;
    else if (ui == "RAW10")
      return IS_CM_SENSOR_RAW10;
    else if (ui == "RAW12")
      return IS_CM_SENSOR_RAW12;
    else if (ui == "RAW16")
      return IS_CM_SENSOR_RAW16;
    else if (ui == "MONO8")
      return IS_CM_MONO8;
    else if (ui == "MONO10")
      return IS_CM_MONO10;
    else if (ui == "MONO12")
      return IS_CM_MONO12;
    else if (ui == "MONO16")
      return IS_CM_MONO16;
    else
      return IS_CM_SENSOR_RAW8;
  }

  int CameraIds::get_trigger_mode(std::string ui)
  {
    if (ui == "Software")
      return IS_SET_TRIGGER_SOFTWARE;
    else if (ui == "HardwareHiLo")
      return IS_SET_TRIGGER_HI_LO;
    else if (ui == "HardwareLoHi")
      return IS_SET_TRIGGER_LO_HI;
    else
      return IS_SET_TRIGGER_OFF;
  }
}