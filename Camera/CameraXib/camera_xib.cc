/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

#include <utils.hh>
#include <camera_exception.hh>
#include <iostream>

#include "camera_xib.hh"

#include <chrono>

namespace camera
{
  CameraXib::CameraXib()
    : Camera("xiq.ini")
    , device_(nullptr)
  {
    name_ = "xiB-64";

	DWORD devices = 0;
	auto res = xiGetNumberDevices(&devices);

    load_default_params();
    if (ini_file_is_open())
      load_ini_params();

	if (ini_file_is_open())
		ini_file_.close();

    frame_.size = sizeof(XI_IMG);
    frame_.bp = nullptr;
    frame_.bp_size = 0;

    // Load functions from CameraUtils.dll.
    load_utils();
  }

  void CameraXib::init_camera()
  {

    auto status = xiOpenDevice(0, &device_);
    if (status != XI_OK)
      throw CameraException(CameraException::NOT_INITIALIZED);

    /* Configure the camera API with given parameters. */
    bind_params();
  }

  void CameraXib::start_acquisition()
  {
    if (xiStartAcquisition(device_) != XI_OK)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
  }

  void CameraXib::stop_acquisition()
  {
    if (xiStopAcquisition(device_) != XI_OK)
      throw CameraException(CameraException::CANT_STOP_ACQUISITION);
  }

  void CameraXib::shutdown_camera()
  {
    if (xiCloseDevice(device_) != XI_OK)
      throw CameraException(CameraException::CANT_SHUTDOWN);
  }

  void* CameraXib::get_frame()
  {
    xiGetImage(device_, FRAME_TIMEOUT, &frame_);

    return frame_.bp;
  }

  void CameraXib::load_default_params()
  {
    /* Fill the frame descriptor. */
    desc_.width = 2048;
    desc_.height = 2048;
    pixel_size_ = 5.5f;
    desc_.depth = 1;
    desc_.byteEndian = Endianness::BigEndian;

    /* Custom parameters. */
    gain_ = 0.f;

    downsampling_rate_ = 1;
    downsampling_type_ = XI_SKIPPING;

    img_format_ = XI_RAW8;

    buffer_policy_ = XI_BP_UNSAFE;

    roi_x_ = 1024;
    roi_y_ = 512;
    roi_width_ = 2048;
    roi_height_ = 2048;

	exposure_time_ = 0;	// free run
  }

  void CameraXib::load_ini_params()
  {
    const boost::property_tree::ptree& pt = get_ini_pt();

    gain_ = pt.get<float>("xiq.gain", gain_);

    downsampling_rate_ = pt.get<unsigned int>("xiq.downsampling_rate", downsampling_rate_);
    // Updating frame size, taking account downsampling.
	desc_.width = desc_.width / static_cast<unsigned short>(downsampling_rate_);
	desc_.height = desc_.height / static_cast<unsigned short>(downsampling_rate_);

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

    {
      const int tmp_roi_x = pt.get<int>("xiq.roi_x", roi_x_);
      const int tmp_roi_y = pt.get<int>("xiq.roi_y", roi_y_);
      const int tmp_roi_width = pt.get<int>("xiq.roi_width", roi_width_);
      const int tmp_roi_height = pt.get<int>("xiq.roi_height", roi_height_);

      /* Making sure ROI settings are valid.
       * Keep in mind that ROI area can't be larger than the
       * initial frame's area (after downsampling!). */
      if (tmp_roi_width > 0 &&
        tmp_roi_height > 0 &&
        tmp_roi_x < desc_.width &&
        tmp_roi_y < desc_.height &&
        tmp_roi_width <= desc_.width &&
        tmp_roi_height <= desc_.height)
      {
        roi_x_ = tmp_roi_x;
        roi_y_ = tmp_roi_y;
        roi_width_ = tmp_roi_width;
        roi_height_ = tmp_roi_height;

        // Don't forget to update the frame descriptor!
        desc_.width = static_cast<unsigned short>(roi_width_);
		desc_.height = static_cast<unsigned short>(roi_height_);
      }
      else
        std::cerr << "[CAMERA] Invalid ROI settings, ignoring ROI." << std::endl;
    }

    trigger_src_ = (XI_TRG_SOURCE)pt.get<unsigned long>("xiq.trigger_src", XI_TRG_OFF);

    exposure_time_ = pt.get<float>("xiq.exposure_time", exposure_time_);
  }

  void CameraXib::bind_params()
  {
    XI_RETURN status = XI_OK;

    const unsigned int name_buffer_size = 32;
    char name[name_buffer_size];

    status |= xiGetParamString(device_, XI_PRM_DEVICE_NAME, &name, name_buffer_size);

    status |= xiSetParamInt(device_, XI_PRM_DOWNSAMPLING, downsampling_rate_);
    status |= xiSetParamInt(device_, XI_PRM_DOWNSAMPLING_TYPE, downsampling_type_);
    status |= xiSetParamInt(device_, XI_PRM_IMAGE_DATA_FORMAT, img_format_);
    status |= xiSetParamInt(device_, XI_PRM_OFFSET_X, roi_x_);
    status |= xiSetParamInt(device_, XI_PRM_OFFSET_Y, roi_y_);
    status |= xiSetParamInt(device_, XI_PRM_WIDTH, roi_width_);
    status |= xiSetParamInt(device_, XI_PRM_HEIGHT, roi_height_);

    status |= xiSetParamInt(device_, XI_PRM_BUFFER_POLICY, buffer_policy_);

	if (exposure_time_)
		status |= xiSetParamFloat(device_, XI_PRM_EXPOSURE, 1.0e6f * exposure_time_);
	else
		status |= xiSetParamFloat(device_, XI_PRM_ACQ_TIMING_MODE, XI_ACQ_TIMING_MODE_FREE_RUN);

    status |= xiSetParamFloat(device_, XI_PRM_GAIN, gain_);

    status |= xiSetParamInt(device_, XI_PRM_TRG_SOURCE, trigger_src_);

    if (status != XI_OK)
      throw CameraException(CameraException::CANT_SET_CONFIG);

    /* Update the frame descriptor. */
    if (img_format_ == XI_RAW16 || img_format_ == XI_MONO16)
      desc_.depth = 2;

    name_ = std::string(name);
  }

  ICamera* new_camera_device()
  {
    return new CameraXib();
  }
}