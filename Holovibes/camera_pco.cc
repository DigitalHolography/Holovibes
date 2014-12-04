#include "camera_pco.hh"
#include "camera_exception.hh"
#include <boost/lexical_cast.hpp>

#include <PCO_err.h>
#include <PCO_errt.h>

#define PCO_RECSTATE_RUN 0x0001
#define PCO_RECSTATE_STOP 0x0000

namespace camera
{
  CameraPCO::CameraPCO(
    const std::string& ini_filepath,
    WORD camera_type)
    : Camera(ini_filepath)
    , device_(nullptr)
    , camera_type_(camera_type)
    , buffers_events_()
    , buffers_()
  {
    for (unsigned int i = 0;
      i < buffers_events_.size();
      ++i)
    {
      std::string event_name = "buffer" + boost::lexical_cast<std::string>(i);
      buffers_events_[i] = CreateEvent(NULL, FALSE, FALSE, event_name.c_str());
    }

    for (unsigned int i = 0;
      i < buffers_.size();
      ++i)
    {
      buffers_[i] = nullptr;
    }
  }

  CameraPCO::~CameraPCO()
  {
    /* Ensure that the camera is closed in case of exception. */
    shutdown_camera();

    for (unsigned int i = 0;
      i < buffers_events_.size();
      ++i)
    {
      CloseHandle(buffers_events_[i]);
    }
  }

  void CameraPCO::init_camera()
  {
    if (PCO_OpenCamera(&device_, 0) != PCO_NOERROR)
      throw CameraException(name_, CameraException::NOT_INITIALIZED);
    
    /* TODO check type*/

    /* Ensure that the camera is not in recording state. */
    stop_acquisition();

    bind_params();

    int status = PCO_NOERROR;
    /* TODO ERROR CHECKING */
    /* Retrieve frame resolution. */
    status |= get_sensor_sizes();
    /* Buffer memory allocation. */
    status |= allocate_buffers();

    if (status != PCO_NOERROR)
      throw CameraException(name_, CameraException::NOT_INITIALIZED);
  }

  void CameraPCO::start_acquisition()
  {
    int status = PCO_NOERROR;

    status |= PCO_ArmCamera(device_);
    status |= PCO_SetRecordingState(device_, PCO_RECSTATE_RUN);
    for (unsigned int i = 0;
      i < buffers_.size();
      ++i)
    {
      status |= PCO_AddBufferEx(
        device_,
        0, 0, i,
        actual_res_x_,
        actual_res_y_,
        16);
    }

    if (status != PCO_NOERROR)
      throw CameraException(name_, CameraException::CANT_START_ACQUISITION);
  }

  void CameraPCO::stop_acquisition()
  {
    PCO_SetRecordingState(device_, PCO_RECSTATE_STOP);
  }

  void CameraPCO::shutdown_camera()
  {
    /* No error checking because that method is called in destructor. */
    PCO_CancelImages(device_);
    PCO_RemoveBuffer(device_);

    for (unsigned int i = 0;
      i < buffers_.size();
      ++i)
    {
      PCO_FreeBuffer(device_, i);
      buffers_[i] = nullptr;
    }
    PCO_CloseCamera(device_);
  }

  void* CameraPCO::get_frame()
  {
    DWORD event_status;
    if ((event_status = WaitForMultipleObjects(
      buffers_events_.size(),
      buffers_events_._Elems,
      FALSE,
      FRAME_TIMEOUT)) < WAIT_ABANDONED_0)
    {
      DWORD buffer_index = event_status - WAIT_OBJECT_0;
      PCO_AddBufferEx(
        device_,
        0, 0, buffer_index,
        actual_res_x_,
        actual_res_y_,
        16);

      return buffers_[buffer_index];
    }

    throw CameraException(name_, CameraException::CANT_GET_FRAME);
  }

  int CameraPCO::get_sensor_sizes()
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
  
  int CameraPCO::allocate_buffers()
  {
    DWORD buffer_size = desc_.width * desc_.height * sizeof(WORD);
    int status = PCO_NOERROR;

    for (unsigned int i = 0;
      i < buffers_.size();
      ++i)
    {
      SHORT buffer_nbr = -1;
      status |= PCO_AllocateBuffer(
        device_,
        &buffer_nbr,
        buffer_size,
        &buffers_[i],
        &buffers_events_[i]);

      assert(buffer_nbr == i);
    }

    return status;
  }
}