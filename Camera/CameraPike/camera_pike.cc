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

#include <camera_exception.hh>

#include "camera_pike.hh"

#define MAXNAMELENGTH 64
#define MAXCAMERAS 1
#define FRAMETIMEOUT 1000

namespace camera
{
  void CameraPike::init_camera()
  {
    unsigned long result;
    FGNODEINFO nodes_info[MAXCAMERAS];
    unsigned long max_nodes = MAXCAMERAS;
    unsigned long copied_nodes = 0;

    // Prepare the entire library for use
    if (FGInitModule(NULL) == FCE_NOERROR)
    {
      /* Retrieve list of connected nodes (cameras)
       * Ask for a maximum number of nodes info to fill (max_nodes)
       * and put them intos nodes_info. It also puts the number of nodes
       * effectively copied into copied_nodes. */
      result = FGGetNodeList(nodes_info, max_nodes, &copied_nodes);

      if (result == FCE_NOERROR && copied_nodes != 0)
      {
        // Connect first node with our cam_ object
        result = cam_.Connect(&nodes_info[0].Guid);
        name_ = get_name_from_device();
        bind_params();
      }
      else
        throw CameraException(CameraException::NOT_CONNECTED);
    }
    else
      throw CameraException(CameraException::NOT_INITIALIZED);
  }

  void CameraPike::start_acquisition()
  {
    // Allocate DMA for the camera
    if (cam_.OpenCapture() != FCE_NOERROR)
      throw CameraException(CameraException::CANT_START_ACQUISITION);

    if (cam_.StartDevice() != FCE_NOERROR)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
  }

  void CameraPike::stop_acquisition()
  {
    if (cam_.StopDevice() != FCE_NOERROR)
      throw CameraException(CameraException::CANT_STOP_ACQUISITION);
  }

  void CameraPike::shutdown_camera()
  {
    // Free all image buffers and close the capture logic
    if (cam_.CloseCapture() != FCE_NOERROR)
      throw CameraException(CameraException::CANT_SHUTDOWN);
  }

  void* CameraPike::get_frame()
  {
    if (cam_.GetFrame(&fgframe_, FRAME_TIMEOUT) == FCE_NOERROR)
    {
      // Put the frame back to DMA
      cam_.PutFrame(&fgframe_);
    }

    return fgframe_.pData;
  }

  std::string CameraPike::get_name_from_device()
  {
    char ccam_name[MAXNAMELENGTH];

    if (cam_.GetDeviceName(ccam_name, MAXNAMELENGTH) != 0)
      return "unknown type";

    return std::string(ccam_name);
  }

  void CameraPike::load_default_params()
  {
    desc_.width = 2048;
    desc_.height = 2048;
    desc_.depth = 1.f;
    pixel_size_ = 7.4f;
    desc_.byteEndian = Endianness::BigEndian;

    subsampling_ = 0;
    gain_ = 0;
    brightness_ = 0;
    exposure_time_ = 1000.0f;
    gamma_ = 0;
    speed_ = 800;

    roi_startx_ = 0;
    roi_starty_ = 0;
    roi_width_ = 2048;
    roi_height_ = 2048;

    trigger_on_ = 0;
    trigger_pol_ = 0;
    trigger_mode_ = 0;
  }

  void CameraPike::load_ini_params()
  {
    const boost::property_tree::ptree& pt = get_ini_pt();

    desc_.width = pt.get<unsigned short>("pike.sensor_width", desc_.width);
    desc_.height = pt.get<unsigned short>("pike.sensor_height", desc_.height);
    desc_.depth = static_cast<float>(pt.get<unsigned short>("pike.bit_depth", 8) + 7) / 8.f;

    subsampling_ = pt.get<int>("pike.subsampling", subsampling_);
    gain_ = pt.get<unsigned long>("pike.gain", gain_);
    brightness_ = pt.get<unsigned long>("pike.brightness", brightness_);
    gamma_ = pt.get<unsigned long>("pike.gamma", gamma_);
    speed_ = pt.get<unsigned long>("pike.speed", speed_);

    roi_startx_ = pt.get<int>("pike.roi_startx", roi_startx_);
    roi_starty_ = pt.get<int>("pike.roi_starty", roi_starty_);
    roi_width_ = pt.get<int>("pike.roi_width", roi_width_);
    roi_height_ = pt.get<int>("pike.roi_height", roi_height_);

    trigger_on_ = pt.get<unsigned long>("pike.trigger_on", trigger_on_);
    trigger_pol_ = pt.get<unsigned long>("pike.trigger_pol", trigger_pol_);
    trigger_mode_ = pt.get<unsigned long>("pike.trigger_mode", trigger_mode_);

    exposure_time_ = pt.get<float>("pike.shutter_time", exposure_time_);
  }

  void CameraPike::bind_params()
  {
    unsigned long status = FCE_NOERROR;

    status |= cam_.SetParameter(FGP_IMAGEFORMAT, to_dcam_format());
    status |= cam_.SetParameter(FGP_GAIN, gain_);
    status |= cam_.SetParameter(FGP_BRIGHTNESS, brightness_);
    status |= cam_.SetParameter(FGP_SHUTTER, static_cast<unsigned long>(exposure_time_));
    status |= cam_.SetParameter(FGP_GAMMA, gamma_);
    status |= cam_.SetParameter(FGP_PHYSPEED, to_speed());
    status |= cam_.SetParameter(FGP_TRIGGER, MAKETRIGGER(trigger_on_, trigger_pol_, 0, trigger_mode_, 0));
    status |= cam_.SetParameter(FGP_XPOSITION, roi_startx_);
    status |= cam_.SetParameter(FGP_YPOSITION, roi_starty_);
    status |= cam_.SetParameter(FGP_XSIZE, roi_width_);
    status |= cam_.SetParameter(FGP_YSIZE, roi_height_);

    if (status != FCE_NOERROR)
      throw CameraException(CameraException::CANT_SET_CONFIG);
  }

  unsigned long CameraPike::to_dcam_format() const
  {
    int mode = 0;
    int color_mode = desc_.depth == 2 ? CM_Y16 : CM_Y8;

    if (desc_.width == 2048 && desc_.height == 2048)
      mode = 0;
    else if (desc_.width == 1024 && desc_.height == 1024)
    {
      if (subsampling_ == 1)
        mode = 3;
      else
        mode = 6;
    }
    else
      mode = 0;

    return MAKEDCAMFORMAT(7, mode, color_mode);
  }

  unsigned long CameraPike::to_speed() const
  {
    if (speed_ == 100)
      return FG_PHYSPEED::PS_100MBIT;
    else if (speed_ == 200)
      return FG_PHYSPEED::PS_200MBIT;
    else if (speed_ == 400)
      return FG_PHYSPEED::PS_400MBIT;
    else if (speed_ == 800)
      return FG_PHYSPEED::PS_800MBIT;
    else
      return FG_PHYSPEED::PS_800MBIT;
  }

  ICamera* new_camera_device()
  {
    return new CameraPike();
  }
}