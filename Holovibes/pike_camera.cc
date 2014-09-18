#include "pike_camera.hh"

#define MAXNAMELENGTH 256
#define MAXCAMERAS 1

namespace camera
{

  bool PikeCamera::init_camera()
  {
    unsigned long result;
    FGNODEINFO* nodes_info;
    unsigned long max_nodes = MAXCAMERAS;
    unsigned long copied_nodes = 0;

    //Prepare the entire library for use
    result = FGInitModule(NULL);

    //FCE_NOERROR = 0
    if (result == FCE_NOERROR)
    {
      //Retrieve list of connected nodes (cameras)
      //Ask for a maximum number of nodes info to fill (max_nodes)
      //and put them intos nodes_info. It also puts the number of nodes
      //effectively copied.
      result = FGGetNodeList(nodes_info, max_nodes, &copied_nodes);
    }

    //If there is no errors and at least one node detected.
    if (result == FCE_NOERROR && copied_nodes)
    {
      //FIXME
      //Connect first node with our cam_ object
    }

    //Retrieve name from device and fill name_ with it
    name_ = get_name_from_device();

    return true;
  }

  void PikeCamera::start_acquisition()
  {
  }

  void PikeCamera::stop_acquisition()
  {

  }

  void PikeCamera::shutdown_camera()
  {

  }

  std::string PikeCamera::get_name_from_device()
  {
    char* ccam_name = new char[MAXNAMELENGTH];

    if (cam_.GetDeviceName(ccam_name, MAXNAMELENGTH) != 0)
      return "";

    std::string cam_name(ccam_name);

    return cam_name;
  }
}