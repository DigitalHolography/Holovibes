#include "pike_camera.hh"

#define MAXNAMELENGTH 256
#define MAXCAMERAS 1
#define FRAMETIMEOUT 1000

namespace camera
{

  bool PikeCamera::init_camera()
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

    return result == FCE_NOERROR && copied_nodes != 0;
  }

  void PikeCamera::start_acquisition()
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

  void PikeCamera::stop_acquisition()
  {

  }

  void PikeCamera::shutdown_camera()
  {

  }

  void* PikeCamera::get_frame()
  {
    FGFRAME fgframe;
    unsigned long result;

    // Retreiving the frame
    result = cam_.GetFrame(&fgframe, FRAMETIMEOUT);

    if (result == FCE_NOERROR)
    {
      // Put the frame back to DMA
      result = cam_.PutFrame(&fgframe);

      std::cout << "Frame received length:"
        << fgframe.Length << " id:"
        << fgframe.Id << std::endl;
    }

    return &fgframe;
  }

  std::string PikeCamera::get_name_from_device()
  {
    char* ccam_name = new char[MAXNAMELENGTH];

    if (cam_.GetDeviceName(ccam_name, MAXNAMELENGTH) != 0)
      return "unknown type";

    std::string cam_name(ccam_name);
    delete ccam_name;
    return cam_name;
  }
}