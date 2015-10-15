#include <CiApi.h>

// DEBUG
#include <iostream>
// DEBUG
#include <cmath>
#include <cstdlib>

#include "camera_adimec.hh"
#include "camera_exception.hh"

namespace camera
{
  // Anonymous namespace allows translation-unit visibility (like static).
  namespace
  {
    /* The Adimec camera returns images with values on 12 bits, each pixel
    ** being encoded on two bytes (16 bits). However, we need to shift each
    ** value towards the 4 unused bits, otherwise the image is very dark.
    ** Example : A pixel's value is : 0x0AF5
    **           We should make it  : 0xAF50
    */
    void update_image(void* buffer)
    {
      const unsigned shift_step = 4;
      size_t* it = reinterpret_cast<size_t*>(buffer);

      for (unsigned y = 0; y < 1440; ++y)
      {
        for (unsigned x = 0; x < 360; ++x)
        {
          it[x + y * 360] <<= shift_step;
        }
      }
    }
  }

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
    desc_.depth = 2.f;
    desc_.endianness = LITTLE_ENDIAN;
    // TODO : Find pixel size.

    load_default_params();
    if (ini_file_is_open())
      load_ini_params();
  }

  CameraAdimec::~CameraAdimec()
  {
    // Make sure the camera is closed at program exit.
    shutdown_camera();
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
    {
      // Camera could not be opened.
      throw CameraException(CameraException::NOT_INITIALIZED);
    }

    bind_params();
  }

  void CameraAdimec::start_acquisition()
  {
    /* First, configuring the camera through CiCamOpen
    ** (the configuration file should be in
    ** BitFlow SDK 6.10\Config\Ctn\
    */
    PCHAR config = "adimec_test_roi.bfml";
    BFRC status = CiCamOpen(board_, config, &camera_);
    if (status != CI_OK)
    {
      std::cerr << "[CAMERA] Could not open cam object\n";
      CiBrdClose(board_);
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    status = CiBrdCamSetCur(board_, camera_, 0);
    if (status != CI_OK)
    {
      std::cerr << "[CAMERA] Could not set cam object\n";
      CiBrdClose(board_);
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    /* Now, allocating buffer(s) for acquisition.
    */
    // We get the frame size (width * height * depth).
    BFU32 size;
    if (CiBrdInquire(board_, CiCamInqFrameSize0, &size) != CI_OK)
    {
      std::cerr << "[CAMERA] Could not get frame size\n";
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    }

    // Aligned allocation ensures fast memory transfers.
    buffer_ = _aligned_malloc(size, 4096);
    if (!buffer_)
    {
      shutdown_camera();
      throw CameraException(CameraException::MEMORY_PROBLEM);
    }
    memset(buffer_, 0, size);

    status = CiAqSetup(board_,
      buffer_,
      size,
      0, // We let the SDK calculate the pitch itself.
      CiDMADataMem, // We don't care about this parameter, it is for another board.
      CiLutBypass, /* TODO : Check that the Cyton board really does not care about this, */
      CiLut12Bit, /* and that we can safely assume that LUTs parameters are ignored.    */
      quad_bank_, // We use a single buffer, in a single bank.
      TRUE,
      CiQTabModeOneBank,
      AqEngJ // We dont' care about this parameter, it is for another board.
      );
    if (status != CI_OK)
    {
      std::cerr << "[CAMERA] Could not setup board for acquisition" << status << std::endl;
      _aligned_free(buffer_);
      CiCamClose(board_, camera_);
      shutdown_camera();
      throw CameraException(CameraException::CANT_START_ACQUISITION);
    }
  }

  void CameraAdimec::stop_acquisition()
  {
    /* Free resources taken by CiAqSetup, in a single function call.
    ** However, the allocated buffer has to be freed manually.
    */
    _aligned_free(buffer_);
    CiCamClose(board_, camera_);
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
      // TODO : Write a logger for missed images.
      std::cerr << "[CAMERA] Could not get frame" << std::endl;
      /*delete[] buffer_;
      CiAqCleanUp(board_, AqEngJ);
      shutdown_camera();
      throw CameraException(CameraException::CANT_GET_FRAME);*/
    }

    update_image(buffer_);
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