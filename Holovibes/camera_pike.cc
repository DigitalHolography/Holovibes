#include "stdafx.h"
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
      ** Ask for a maximum number of nodes info to fill (max_nodes)
      ** and put them intos nodes_info. It also puts the number of nodes
      ** effectively copied into copied_nodes.
      */
      result = FGGetNodeList(nodes_info, max_nodes, &copied_nodes);

      if (result == FCE_NOERROR && copied_nodes != 0)
      {
        // Connect first node with our cam_ object
        result = cam_.Connect(&nodes_info[0].Guid);
        name_ = get_name_from_device();
        bind_params();
      }
      else
        throw new CameraException(name_, CameraException::camera_error::NOT_CONNECTED);
    }
    else
      throw new CameraException(name_, CameraException::camera_error::NOT_INITIALIZED);
  }

  void CameraPike::start_acquisition()
  {
    // Allocate DMA for the camera
    if(cam_.OpenCapture() != FCE_NOERROR)
      throw new CameraException(name_, CameraException::camera_error::CANT_START_ACQUISITION);

    if (cam_.StartDevice() != FCE_NOERROR)
      throw new CameraException(name_, CameraException::camera_error::CANT_START_ACQUISITION);
  }

  void CameraPike::stop_acquisition()
  {
    if(cam_.StopDevice() != FCE_NOERROR)
      throw new CameraException(name_, CameraException::camera_error::CANT_STOP_ACQUISITION);
  }

  void CameraPike::shutdown_camera()
  {
    // Free all image buffers and close the capture logic
    if(cam_.CloseCapture() != FCE_NOERROR)
      throw new CameraException(name_, CameraException::camera_error::CANT_SHUTDOWN);
  }

  void* CameraPike::get_frame()
  {
    if (cam_.GetFrame(&fgframe_, FRAMETIMEOUT) == FCE_NOERROR)
    {
      // Put the frame back to DMA
      cam_.PutFrame(&fgframe_);

      /*std::cout << "Frame received length:"
        << fgframe_.Length << " id:"
        << fgframe_.Id << std::endl;*/
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
    desc_.width = 1600;
    desc_.height = 1200;
    desc_.bit_depth = 8;
    desc_.endianness = LITTLE_ENDIAN;
  }

  void CameraPike::load_ini_params()
  {
    const boost::property_tree::ptree& pt = get_ini_pt();

    desc_.width = pt.get<int>("pike.sensor_width", 2048);
    desc_.height = pt.get<int>("pike.sensor_height", 2048);
    desc_.bit_depth = pt.get<int>("pike.bit_depth", 8);
    desc_.endianness;
    desc_.pixel_size;

    exposure_time_ = pt.get<float>("pike.exposure_time", exposure_time_);
    subsampling_ = pt.get<int>("pike.subsampling", 0);
  }

  void print_info(FGPINFO pinfo)
  {
    std::cout << "Min: " << IMGRES(pinfo.MinValue) << " Max: " << IMGRES(pinfo.MaxValue) << std::endl;
    std::cout << "Actual: " << IMGRES(pinfo.IsValue) << std::endl;
  }

  void CameraPike::bind_params()
  {
    // Status
    unsigned long status = FCE_NOERROR;

    // Image format
    status = cam_.SetParameter(FGP_IMAGEFORMAT, to_dcam_format());

    if (status != FCE_NOERROR)
      throw new CameraException(name_, CameraException::camera_error::CANT_SET_CONFIG);
  }

  unsigned long CameraPike::to_dcam_format()
  {
    int mode = 0;
    int color_mode = desc_.bit_depth == 16 ? CM_Y16 : CM_Y8;

    if (desc_.width == 2048 && desc_.height == 2048)
      mode = 0;
    else if (desc_.width == 1024 && desc_.height == 1024)
    {
      if (subsampling_ == 1)
        mode == 3;
      else
        mode == 6;
    }
    else
      mode = 0;

    return MAKEDCAMFORMAT(7, mode, color_mode);
  }
}