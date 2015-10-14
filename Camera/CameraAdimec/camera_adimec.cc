#include <CiApi.h>
// DEBUG
#include <R64Api.h>
#include <iostream>
#include <cmath>
#include <cstdlib>

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
    , board_ { nullptr }
  , buffer_ { nullptr }
  , quad_bank_ { BFQTabBank0 }
  {
    name_ = "Adimec";
    desc_.width = 1440;
    desc_.height = 1440;
    desc_.depth = 1.5f;
    desc_.endianness = LITTLE_ENDIAN;
    // TODO : Find pixel size.

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
    CiENTRY entry;

    BFRC status = CiSysBrdFind(type, number, &entry);
    if (status != CI_OK)
      // No board was found.
      throw CameraException(CameraException::NOT_CONNECTED);

    status = CiBrdOpen(&entry, &board_, CiSysInitialize);
    if (status != CI_OK)
      // Camera could not be opened.
      throw CameraException(CameraException::NOT_INITIALIZED);

    bind_params();
  }

  void CameraAdimec::start_acquisition()
  {
    const unsigned buffer_pitch = static_cast<unsigned>(ceil(desc_.width * desc_.depth));
    buffer_ = new char[buffer_pitch * buffer_pitch];
    if (!buffer_)
    {
      shutdown_camera();
      throw CameraException(CameraException::MEMORY_PROBLEM);
    }
    memset(buffer_, 0, buffer_pitch * buffer_pitch);
    // DEBUG
    char* end = reinterpret_cast<char*>(buffer_)+((buffer_pitch * buffer_pitch) - 1);
    std::cout << "Buffer ranges from " << buffer_ << " to " << (void*)end << std::endl;
    // ! DEBUG

    BFRC status = CiAqSetup(board_,
      buffer_,
      buffer_pitch * buffer_pitch,
      buffer_pitch,
      CiDMADataMem, // We don't care about this parameter, it is for another board.
      CiLutBank1, /* TODO : Check that the Cyton board really does not care about this, */
      CiLut12Bit, /* and that we can safely assume that LUTs parameters are ignored.    */
      quad_bank_, // We use a single buffer, in a single bank.
      TRUE,
      CiQTabModeOneBank,
      AqEngJ // We dont' care about this parameter, it is for another board.
      );
    if (status != CI_OK)
    {
      std::cerr << "[CAMERA] Could not setup board for acquisition" << status << std::endl;
      delete[] buffer_;
      shutdown_camera();
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    }
  }

  void CameraAdimec::stop_acquisition()
  {
    /* Free resources taken by CiAqSetup, in a single function call.
    ** However, the allocated buffer has to be freed manually.
    */
    delete[] buffer_;
    if (CiAqCleanUp(board_, AqEngJ) != CI_OK)
    {
      std::cerr << "[CAMERA] Could not stop acquisition cleanly." << std::endl;
      shutdown_camera();
      throw CameraException(CameraException::CANT_STOP_ACQUISITION);
    }
  }

  void CameraAdimec::shutdown_camera()
  {
    CiBrdClose(board_);
  }

  void* CameraAdimec::get_frame()
  {
    BFRC status = CiAqCommand(board_,
      CiConSnap,
      CiConWait,
      quad_bank_,
      AqEngJ);
    if (status != CI_OK)
    {
      std::cerr << "[CAMERA] Could not get frame" << std::endl;
      /*delete[] buffer_;
      CiAqCleanUp(board_, AqEngJ);
      shutdown_camera();
      throw CameraException(CameraException::CANT_GET_FRAME);*/
    }
    else // DEBUG : Remove me later
      std::cout << "\nAcquired image properly" << std::endl;

    return buffer_;
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