#include "camera_edge.hh"
#include "camera_exception.hh"

#include <PCO_err.h>

#define PCO_RECSTATE_RUN 0x0001
#define PCO_RECSTATE_STOP 0x0000

namespace camera
{
  CameraEdge::CameraEdge()
    : Camera("edge.ini")
    , device_(nullptr)
    , refresh_event_(CreateEvent(NULL, FALSE, FALSE, "ImgReady"))
    , buffer_(nullptr)
  {
    name_ = "edge";
    load_default_params();
    if (ini_file_is_open())
      load_ini_params();
  }

  CameraEdge::~CameraEdge()
  {
    /* Ensure that the camera is closed in case of exception. */
    shutdown_camera();
    CloseHandle(refresh_event_);
  }

  void CameraEdge::init_camera()
  {
    PCO_CameraType type;

    if (PCO_OpenCamera(&device_, 0) != PCO_NOERROR)
      throw CameraException(name_, CameraException::NOT_INITIALIZED);

    /* Ensure that the camera is not in recording state. */
    stop_acquisition();

    bind_params();
    if (pco_allocate_buffer() != PCO_NOERROR)
      throw CameraException(name_, CameraException::MEMORY_PROBLEM);
  }

  void CameraEdge::start_acquisition()
  {
    int status = PCO_NOERROR;

    status |= PCO_SetRecordingState(device_, PCO_RECSTATE_RUN);
    status |= PCO_AddBufferEx(device_, 0, 0, 0, static_cast<WORD>(desc_.width), static_cast<WORD>(desc_.height), 16);

    if (status != PCO_NOERROR)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
  }

  void CameraEdge::stop_acquisition()
  {
    PCO_SetRecordingState(device_, PCO_RECSTATE_STOP);
  }

  void CameraEdge::shutdown_camera()
  {
    /* No error checking because that method is called in destructor. */
    PCO_CancelImages(device_);
    PCO_RemoveBuffer(device_);
    PCO_FreeBuffer(device_, 0);
    PCO_CloseCamera(device_);
  }

  void* CameraEdge::get_frame()
  {
    if (WaitForSingleObject(refresh_event_, 2000) == WAIT_OBJECT_0)
    {
      PCO_AddBufferEx(device_, 0, 0, 0, static_cast<WORD>(actual_res_x_), static_cast<WORD>(actual_res_y_), 16);
      return buffer_;
    }
    throw CameraException(name_, CameraException::CANT_GET_FRAME);
  }

  void CameraEdge::load_default_params()
  {
    exposure_time_ = 0.05f;
    extended_sensor_format_ = true;
    pixel_rate_ = 110;
    binning_ = false;

    /* Fill frame descriptor const values. */
    desc_.depth = 2;
    desc_.endianness = LITTLE_ENDIAN;
    desc_.pixel_size = 6.45f;
    desc_.width = 2048;
    desc_.height = 2048;
  }

  void CameraEdge::load_ini_params()
  {
    const boost::property_tree::ptree& pt = get_ini_pt();

    exposure_time_ = pt.get<float>("pixelfly.exposure_time", exposure_time_);
    extended_sensor_format_ = pt.get<bool>("pixelfly.extended_sensor_format", extended_sensor_format_);
    pixel_rate_ = pt.get<unsigned int>("pixelfly.pixel_rate", pixel_rate_);
    if (pixel_rate_ != 12 || pixel_rate_ != 25)
      pixel_rate_ = 12;
    binning_ = pt.get<bool>("pixelfly.binning", binning_);
  }

  void CameraEdge::bind_params()
  {
    int status = PCO_NOERROR;

    status |= PCO_SetSensorFormat(device_, extended_sensor_format_);
    status |= PCO_SetPixelRate(device_, static_cast<DWORD>(pixel_rate_ * 1e6));
    {
      WORD binning_x = 1;
      WORD binning_y = 1;

      if (binning_)
      {
        binning_x = 2;
        binning_y = 2;
      }

      status |= PCO_SetBinning(device_, binning_x, binning_y);
    }
    {
      /* Convert exposure time in milliseconds. */
      exposure_time_ *= 1e3;

      /* base_time : 0x0002 = ms, 0x0001 = us, 0x0000 = ns */
      WORD base_time;

      for (base_time = 0x0002; base_time > 0 && exposure_time_ < 1.0f; --base_time)
        exposure_time_ *= 1e3;

      status |= PCO_SetDelayExposureTime(device_, 0, static_cast<DWORD>(exposure_time_), 0, base_time);
    }
    status |= PCO_ArmCamera(device_);
    status |= pco_get_sizes();

    if (status != PCO_NOERROR)
      throw CameraException(name_, CameraException::CANT_SET_CONFIG);
  }

  int CameraEdge::pco_get_sizes()
  {
    WORD ccdres_x, ccdres_y;

    int status = PCO_GetSizes(
      device_,
      &actual_res_x_,
      &actual_res_y_,
      &ccdres_x,
      &ccdres_y);

#if _DEBUG
    std::cout << actual_res_x_ << ", " << actual_res_y_ << std::endl;
#endif

    return status;
  }

  int CameraEdge::pco_allocate_buffer()
  {
    SHORT buffer_nbr = -1;
    DWORD buffer_size = desc_.width * desc_.height * sizeof(WORD);

    int status = PCO_AllocateBuffer(
      device_,
      &buffer_nbr,
      buffer_size,
      &buffer_,
      &refresh_event_);

    assert(buffer_nbr == 0);
    return status;
  }
}