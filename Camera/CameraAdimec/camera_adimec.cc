#include <CiApi.h>
// DEBUG
#include <iostream>
#include <cmath>

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
  {
    name_ = "Adimec";
    desc_.width = 1440;
    desc_.height = 1440;
    desc_.depth = 12;
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
    /* Remember that bit depth is in bits, not bytes; thus we have to divide
    ** by 8 appropriately.
    */
    const unsigned buffer_pitch = static_cast<unsigned>(ceil(desc_.width * (desc_.depth / 8.0f)));
    buffer_ = new char[buffer_pitch * buffer_pitch];
    if (!buffer_)
      throw CameraException(CameraException::MEMORY_PROBLEM);

    BFRC status = CiAqSetup(board_,
      buffer_,
      buffer_pitch * buffer_pitch,
      buffer_pitch,
      CiDMADataMem, // We don't care about this parameter, it is for another board.
      CiLutBank0,
      CiLut12Bit,
      CiQTabBank0, // We use a single buffer, in a single bank.
      true,
      CiQTabModeOneBank, // We don't need to use several cameras at the same time.
      AqEngJ // We dont' care about this parameter, it is for another board.
      );
    if (status != CI_OK)
    {
      std::cerr << "[CAMERA] Could not setup board for acquisition." << std::endl;
      delete buffer_;
      CiBrdClose(board_);
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    }
  }

  void CameraAdimec::stop_acquisition()
  {
    /* Free resources taken by CiAqSetup, in a single function call.
    ** However, the allocated buffer has to be freed manually.
    */
    delete buffer_;
    if (CiAqCleanUp(board_, AqEngJ) != CI_OK)
    {
      std::cerr << "[CAMERA] Could not stop acquisition cleanly." << std::endl;
      CiBrdClose(board_);
      throw CameraException(CameraException::CANT_STOP_ACQUISITION);
    }
  }

  void CameraAdimec::shutdown_camera()
  {
    CiBrdClose(board_);
  }

  void* CameraAdimec::get_frame()
  {
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