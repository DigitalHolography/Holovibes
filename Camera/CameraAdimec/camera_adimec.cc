#include <CiApi.h>
// DEBUG
#include <iostream>

#include "camera_adimec.hh"
#include "camera_exception.hh"

namespace camera
{
  ICamera* new_camera_device()
  {
    return new CameraAdimec();
  }

  /* Public interface
  */

  CameraAdimec::CameraAdimec()
    : Camera("adimec.ini")
    , board_{ nullptr }
  {
    name_ = "Adimec";

    load_default_params();
    if (ini_file_is_open())
      load_ini_params();
  }

  CameraAdimec::~CameraAdimec()
  {
  }

  void CameraAdimec::init_camera()
  {
    /* We don't want a specific type of board; there should not
    ** be more than one anyway.
    */
    BFU32 type = CISYS_TYPE_ANY;
    BFU32 number = 0;
    PBFENTRY entry = new _BFENTRY();

    BFRC status = CiSysBrdFind(type, number, entry);
    if (status != CI_OK)
    {
      // No board was found.
      std::cout << "Could not find board" << std::endl;
      throw CameraException(CameraException::NOT_CONNECTED);
    }
    else
    {
      std::cout << "Found board" << std::endl;
    }

    status = CiBrdOpen(entry, &board_, CiSysInitialize);
    if (status != CI_OK)
    {
      // Camera could not be opened.
      std::cout << "Could not open camera" << std::endl;
      throw CameraException(CameraException::NOT_INITIALIZED);
    }
    else
    {
      std::cout << "Opened camera" << std::endl;
    }

    bind_params();
  }

  void CameraAdimec::start_acquisition()
  {
  }

  void CameraAdimec::stop_acquisition()
  {
  }

  void CameraAdimec::shutdown_camera()
  {
  }

  void* CameraAdimec::get_frame()
  {
    return nullptr;
  }

  /* Private methods
  */

  void CameraAdimec::load_default_params()
  {
  }

  void CameraAdimec::load_ini_params()
  {
  }

  void CameraAdimec::bind_params()
  {
  }
}