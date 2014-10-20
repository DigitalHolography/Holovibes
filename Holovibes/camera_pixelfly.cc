#include "stdafx.h"
#include "camera_pixelfly.hh"
#include "camera_exception.hh"

#include <PCO_err.h>

#define PCO_RECSTATE_RUN 0x0001
#define PCO_RECSTATE_STOP 0x0000

namespace camera
{
  CameraPixelfly::CameraPixelfly()
    : Camera("pixelfly.ini")
    , device_(nullptr)
    , refresh_event_(CreateEvent(NULL, FALSE, FALSE, "ImgReady"))
    , buffer_(nullptr)
  {
    load_default_params();
    if (ini_file_is_open())
      load_ini_params();
  }

  CameraPixelfly::~CameraPixelfly()
  {
    /* Ensure that the camera is closed in case of exception. */
    shutdown_camera();
    CloseHandle(refresh_event_);
  }

  void CameraPixelfly::init_camera()
  {
    if (PCO_OpenCamera(&device_, 0) != PCO_NOERROR)
      throw CameraException(name_, CameraException::NOT_INITIALIZED);

    /* Ensure that the camera is not in recording state. */
    stop_acquisition();

    bind_params();
    pco_allocate_buffer();
  }

  void CameraPixelfly::start_acquisition()
  {
    int status = PCO_NOERROR;
    PCO_SetRecordingState(device_, PCO_RECSTATE_RUN);
    if (status != PCO_NOERROR)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
    PCO_AddBufferEx(device_, 0, 0, 0, (WORD)desc_.width, (WORD)desc_.height, 16);
  }

  void CameraPixelfly::stop_acquisition()
  {
    PCO_SetRecordingState(device_, PCO_RECSTATE_STOP);
  }

  void CameraPixelfly::shutdown_camera()
  {
    PCO_CancelImages(device_);
    PCO_RemoveBuffer(device_);
    PCO_FreeBuffer(device_, 0);
    PCO_CloseCamera(device_);
  }

  void* CameraPixelfly::get_frame()
  {
    if (WaitForSingleObject(refresh_event_, 500) == WAIT_OBJECT_0)
    {
      PCO_AddBufferEx(device_, 0, 0, 0, (WORD)desc_.width, (WORD)desc_.height, 16);
      return buffer_;
    }
    return nullptr;
  }

  void CameraPixelfly::load_default_params()
  {
    name_ = "pco.pixelfly";
    exposure_time_ = 0.003f;
    extended_sensor_format_ = false;
    pixel_rate_ = 12;
    binning_ = false;
    ir_sensitivity_ = false;

    /* Fill frame descriptor const values. */
    desc_.bit_depth = 14;
    desc_.endianness = LITTLE_ENDIAN;
    desc_.pixel_size = 6.45f;
  }

  void CameraPixelfly::load_ini_params()
  {
    const boost::property_tree::ptree& pt = get_ini_pt();

    exposure_time_ = pt.get<float>("pixelfly.exposure_time", exposure_time_);
    extended_sensor_format_ = pt.get<bool>("pixelfly.extended_sensor_format", extended_sensor_format_);
    pixel_rate_ = pt.get<unsigned int>("pixelfly.pixel_rate", pixel_rate_);
    if (pixel_rate_ != 12 || pixel_rate_ != 25)
      pixel_rate_ = 12;
    binning_ = pt.get<bool>("pixelfly.binning", binning_);
    ir_sensitivity_ = pt.get<bool>("pixelfly.ir_sensitivity", ir_sensitivity_);
  }

  void CameraPixelfly::bind_params()
  {
    PCO_SetSensorFormat(device_, extended_sensor_format_);
    PCO_SetPixelRate(device_, (DWORD)(pixel_rate_ * 1e6));
    PCO_SetIRSensitivity(device_, ir_sensitivity_);
    {
      WORD binning_x = 1;
      WORD binning_y = 1;

      if (binning_)
      {
        binning_x = 2;
        binning_y = 2;
      }

      PCO_SetBinning(device_, binning_x, binning_y);
    }
    {
      /* Convert exposure time in milliseconds. */
      exposure_time_ *= 1e3;

      /* base_time : 0x0002 = ms, 0x0001 = us, 0x0000 = ns */
      WORD base_time;

      for (base_time = 0x0002; base_time > 0 && exposure_time_ < 1.0f; --base_time)
        exposure_time_ *= 1e3;

      PCO_SetDelayExposureTime(device_, 0, (DWORD)exposure_time_, 0, base_time);
    }
    PCO_ArmCamera(device_);

    pco_get_sizes();
  }

  void CameraPixelfly::pco_get_sizes()
  {
    WORD actualres_x, actualres_y;
    WORD ccdres_x, ccdres_y;

    PCO_GetSizes(
      device_,
      &actualres_x,
      &actualres_y,
      &ccdres_x,
      &ccdres_y);

    /* Fill the frame descriptor width/height fields. */
    desc_.width = actualres_x;
    desc_.height = actualres_y;

#if _DEBUG
    std::cout << actualres_x << ", " << actualres_y << std::endl;
#endif
  }

  void CameraPixelfly::pco_allocate_buffer()
  {
    SHORT buffer_nbr = -1;
    DWORD buffer_size = desc_.width * desc_.height * sizeof(WORD);

    PCO_AllocateBuffer(
      device_,
      &buffer_nbr,
      buffer_size,
      &buffer_,
      &refresh_event_);

    assert(buffer_nbr == 0);
  }
}