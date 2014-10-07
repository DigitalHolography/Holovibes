#include "stdafx.h"
#include "camera_pixelfly.hh"

namespace camera
{
  CameraPixelfly::CameraPixelfly()
    : Camera("pixelfly.ini")
  {
    acquiring_ = false;
    frame_bit_depth_ = 14;
    bining_x_ = 1;
    bining_y_ = 1;
    internal_buff_alloc_ = true;
    pixel_rate_ = 25;
    irSensitivityEnabled_ = false;
    extendedSensorFormatEnabled_ = false;
    error_ = 0;
    refreshEvent_ = NULL;
    refreshEvent_ = CreateEvent(NULL, FALSE, FALSE, NULL);
    inited_ = false;
  }

  void CameraPixelfly::init_camera()
  {
    if (PCO_OpenCamera(&my_cam_, 0) != 0);
#if 0
    // TODO: Fix me
      return false;
#endif
    stop_acquisition();
    set_sensor();
    buff_size = frame_height_ * frame_width_ * 2; // 16bits 2octets per pixel

    short buff_number = -1; //permit to allocate a newbuffer
    buff_alloc();
    error_ = PCO_AllocateBuffer(my_cam_, &buff_number, buff_size, &frame_buffer_, &refreshEvent_);
    check_error(error_, "allocateBUffer");
    inited_ = true;
  }

  void CameraPixelfly::buff_alloc()
  {
    if (internal_buff_alloc_)
      frame_buffer_ = NULL;
    else
      frame_buffer_ = (WORD*)malloc(buff_size); // WORD* is void*
  }

  int CameraPixelfly::get_frame_size()
  {
    if (inited_)
      return (frame_height_ * frame_width_ * 2);
    return 0; // error_checking to place
  }

  void CameraPixelfly::start_acquisition()
  {
    acquiring_ = true;
    error_ = PCO_SetRecordingState(my_cam_, (WORD)0x0001);
    check_error(error_, "set recording state");
    error_ = PCO_AddBufferEx(my_cam_, 0, 0, 0, frame_width_,
      frame_height_, (WORD)frame_bit_depth_);
    check_error(error_, "addbufferEX start");
  }

  void CameraPixelfly::stop_acquisition()
  {
    acquiring_ = false;
    PCO_SetRecordingState(my_cam_, (WORD)0x0000);
  }

  void CameraPixelfly::shutdown_camera()
  {
    PCO_CancelImages(my_cam_);
    PCO_RemoveBuffer(my_cam_);
    PCO_FreeBuffer(&my_cam_, 1);
    PCO_CloseCamera(&my_cam_);
    if (!internal_buff_alloc_)
    {
      free(frame_buffer_);
    }
    CloseHandle(refreshEvent_);
  }

  void* CameraPixelfly::get_frame(void)
  {
    static DWORD dwWaitStatus = 0;
    dwWaitStatus = WaitForSingleObject(refreshEvent_, INFINITE);
    if (dwWaitStatus == WAIT_OBJECT_0)
    {
      error_ = PCO_AddBufferEx(my_cam_,
        (DWORD)0,
        (DWORD)0,
        (SHORT)0,
        (WORD)frame_width_,
        (WORD)frame_height_,
        (WORD)frame_bit_depth_);
      check_error(error_, "addbuffer_ex");
    }
    return frame_buffer_;
  }

  void CameraPixelfly::set_sensor()
  {
    error_ = PCO_SetIRSensitivity(my_cam_,
      (WORD)(irSensitivityEnabled_ ? 0x0001 : 0x0000));
    check_error(error_, "IR");
    error_ = PCO_SetSensorFormat(my_cam_,
      (WORD)(extendedSensorFormatEnabled_ ? 0x0001 : 0x0000));
    check_error(error_, "sensorformat");
    error_ = PCO_SetPixelRate(my_cam_, (DWORD)(pixel_rate_ * 1e6));
    check_error(error_, "pixel rate");
    error_ = PCO_SetTriggerMode(my_cam_, 0x0000);
    check_error(error_, "trigger mode");
    error_ = PCO_SetBinning(my_cam_, bining_x_, bining_y_);
    check_error(error_, "set bining");
    error_ = PCO_ArmCamera(my_cam_);
    check_error(error_, "ArmCamera");
    error_ = PCO_GetSizes(my_cam_, &frame_height_, &frame_width_,
      &max_frame_height_, &max_frame_width_);
    check_error(error_, "getSizes");
  }

  void CameraPixelfly::check_error(DWORD error, std::string msg)
  {
    if (error != 0)
    {
      std::cout << msg << std::endl;
      char *er = (char *)malloc(sizeof (char)* 500);
      DWORD len = 500;
      PCO_GetErrorText(error_, er, len);
      std::cout << er << std::endl;
    }
  }

  void CameraPixelfly::load_default_params()
  {

  }

  void CameraPixelfly::load_ini_params()
  {

  }

  void CameraPixelfly::bind_params()
  {

  }
}