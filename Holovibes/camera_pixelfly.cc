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
    pco_set_size_parameters();
    load_default_params();
    if (ini_file_is_open())
      load_ini_params();
    stop_acquisition();
    shutdown_camera();
  }

  void CameraPixelfly::init_camera()
  {
    if (PCO_OpenCamera(&device_, 0) != PCO_NOERROR)
      throw CameraException(name_, CameraException::NOT_INITIALIZED);

    pco_fill_structures();

    bind_params();
    pco_get_sizes();
    pco_allocate_buffer();
  }

  void CameraPixelfly::start_acquisition()
  {
    int status = PCO_NOERROR;
    status |= PCO_SetRecordingState(device_, PCO_RECSTATE_RUN);
    if (status != PCO_NOERROR)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
    status |= PCO_AddBufferEx(device_, 0, 0, 0, desc_.width, desc_.height, 16);
  }

  void CameraPixelfly::stop_acquisition()
  {
    PCO_CancelImages(device_);
    PCO_SetRecordingState(device_, PCO_RECSTATE_STOP);
  }

  void CameraPixelfly::shutdown_camera()
  {
    PCO_RemoveBuffer(device_);
    PCO_FreeBuffer(device_, 0);
    PCO_CloseCamera(device_);
    CloseHandle(refresh_event_);
  }

  void* CameraPixelfly::get_frame()
  {
    if (WaitForSingleObject(refresh_event_, 500) == WAIT_OBJECT_0)
    {
      assert(PCO_AddBufferEx(device_, 0, 0, 0, desc_.width, desc_.height, 16) == PCO_NOERROR);
      return buffer_;
    }
    return nullptr;
  }

  void CameraPixelfly::load_default_params()
  {}

  void CameraPixelfly::load_ini_params()
  {}

  void CameraPixelfly::bind_params()
  {
    PCO_SetPixelRate(device_, 25 * 1e6);
    PCO_ArmCamera(device_);
  }

  void CameraPixelfly::pco_set_size_parameters()
  {
    pco_general_.wSize = sizeof(pco_general_);
    pco_general_.strCamType.wSize = sizeof(pco_general_.strCamType);
    pco_camtype_.wSize = sizeof(pco_camtype_);
    pco_sensor_.wSize = sizeof(pco_sensor_);
    pco_sensor_.strDescription.wSize = sizeof(pco_sensor_.strDescription);
    pco_sensor_.strDescription2.wSize = sizeof(pco_sensor_.strDescription2);
    pco_description_.wSize = sizeof(pco_description_);
    pco_timing_.wSize = sizeof(pco_timing_);
    pco_storage_.wSize = sizeof(pco_storage_);
    pco_recording_.wSize = sizeof(pco_recording_);
  }

  void CameraPixelfly::pco_fill_structures()
  {
    PCO_GetGeneral(device_, &pco_general_);
    PCO_GetCameraType(device_, &pco_camtype_);
    PCO_GetSensorStruct(device_, &pco_sensor_);
    PCO_GetCameraDescription(device_, &pco_description_);
    PCO_GetTimingStruct(device_, &pco_timing_);
    PCO_GetStorageStruct(device_, &pco_storage_);
    PCO_GetRecordingStruct(device_, &pco_recording_);
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

    desc_.width = actualres_x;
    desc_.height = actualres_y;
    desc_.bit_depth = 14;
    desc_.endianness = LITTLE_ENDIAN;
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