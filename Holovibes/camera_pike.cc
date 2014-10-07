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
    result = FGInitModule(NULL);

    // FCE_NOERROR = 0
    if (result == FCE_NOERROR)
    {
      /* Retrieve list of connected nodes (cameras)
      ** Ask for a maximum number of nodes info to fill (max_nodes)
      ** and put them intos nodes_info. It also puts the number of nodes
      ** effectively copied into copied_nodes.
      */
      result = FGGetNodeList(nodes_info, max_nodes, &copied_nodes);
    }

    // If there is no errors and at least one node detected.
    if (result == FCE_NOERROR && copied_nodes != 0)
    {
      // Connect first node with our cam_ object
      // Connection betzeen real and logical device.
      result = cam_.Connect(&nodes_info[0].Guid);

      // Retrieve name from device and fill name_ with it
      name_ = get_name_from_device();
    }

#if 0
    // TODO: Fix me
    return result == FCE_NOERROR && copied_nodes != 0;
#endif
  }

  void CameraPike::start_acquisition()
  {
    unsigned long result;

    // Allocate DMA for the camera
    result = cam_.OpenCapture();

    if (result == FCE_NOERROR)
    {
      // Starts the image device
      result = cam_.StartDevice();
    }
  }

  void CameraPike::stop_acquisition()
  {
    cam_.StopDevice();
  }

  void CameraPike::shutdown_camera()
  {
    // Free all image buffers and close the capture logic
    cam_.CloseCapture();
  }

  void* CameraPike::get_frame()
  {
    unsigned long result;

    // Retreiving the frame
    result = cam_.GetFrame(&fgframe_, FRAMETIMEOUT);

    if (result == FCE_NOERROR)
    {
      // Put the frame back to DMA
      result = cam_.PutFrame(&fgframe_);

      std::cout << "Frame received length:"
        << fgframe_.Length << " id:"
        << fgframe_.Id << std::endl;
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

  }

  void CameraPike::load_ini_params()
  {

  }

  void CameraPike::bind_params()
  {

  }
}