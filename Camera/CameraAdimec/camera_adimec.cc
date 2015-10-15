#include <CiApi.h>

// DEBUG
#include <iostream>
// DEBUG
#include <cmath>
#include <cstdlib>

#include "camera_adimec.hh"

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
    void update_image(void* buffer, unsigned width, unsigned height)
    {
      const unsigned shift_step = 4;
      size_t* it = reinterpret_cast<size_t*>(buffer);

      for (unsigned y = 0; y < width; ++y)
      {
        for (unsigned x = 0; x < height / 4; ++x)
        {
          it[x + y * height / 4] <<= shift_step;
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
    desc_.width = 512;
    desc_.height = 512;
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

    err_check(CiSysBrdFind(type, number, &entry),
      "No board found.",
      CameraException::NOT_CONNECTED,
      CloseFlag::NO_BOARD);

    err_check(CiBrdOpen(&entry, &board_, CiSysInitialize),
      "Could not open board.",
      CameraException::NOT_INITIALIZED,
      CloseFlag::NO_BOARD);

    bind_params();
  }

  void CameraAdimec::start_acquisition()
  {
    /* First, configuring the camera through CiCamOpen
    ** (the configuration file should be in
    ** BitFlow SDK 6.10\Config\Ctn\
    */
    PCHAR config = "adimec_default.bfml";
    err_check(CiCamOpen(board_, config, &camera_),
      "Could not open cam object",
      CameraException::CANT_START_ACQUISITION,
      CloseFlag::BOARD);

    err_check(CiBrdCamSetCur(board_, camera_, 0),
      "Could not set cam object",
      CameraException::CANT_START_ACQUISITION,
      CloseFlag::BOARD | CloseFlag::CAM);

    /* Setting ROI parameters.
    */
    err_check(CiAqROISet(board_, 0, 0, 512, 512, AqEngJ),
      "Could not set ROI",
      CameraException::CANT_SET_CONFIG,
      CloseFlag::BOARD | CloseFlag::CAM);

    /* Now, allocating buffer(s) for acquisition.
    */
    // We get the frame size (width * height * depth).
    BFU32 size;
    err_check(CiBrdInquire(board_, CiCamInqFrameSize0, &size),
      "Could not get frame size",
      CameraException::CANT_START_ACQUISITION,
      CloseFlag::BOARD | CloseFlag::CAM);
    std::cout << "Frame size : " << size << std::endl;

    // Aligned allocation ensures fast memory transfers.
    buffer_ = _aligned_malloc(size, 4096);
    err_check(buffer_ == 0,
      "Could not allocate buffer memory",
      CameraException::MEMORY_PROBLEM,
      CloseFlag::BOARD | CloseFlag::CAM);
    memset(buffer_, 0, size);

    err_check(CiAqSetup(board_,
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
      ),
      "Could not setup board for acquisition",
      CameraException::CANT_START_ACQUISITION,
      CloseFlag::ALL);
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
    }

    update_image(buffer_, 512, 512);
    return buffer_;
  }

  /* Private methods
  */

  void CameraAdimec::err_check(BFRC status, std::string err_mess, CameraException cam_ex, int flag)
  {
    if (status != CI_OK)
    {
      std::cerr << "[CAMERA] " << err_mess << "\n";

      if (flag & 0x0F0)
        CiCamClose(board_, camera_);
      if (flag & 0xF00)
        _aligned_free(buffer_);
      if (flag & 0x00F)
        CiBrdClose(board_);

      throw cam_ex;
    }
  }

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