#include "camera_pco.hh"
#include <camera_exception.hh>
#include <boost/lexical_cast.hpp>

#include <PCO_err.h>
#include <PCO_errt.h>

#define PCO_RECSTATE_RUN 0x0001
#define PCO_RECSTATE_STOP 0x0000

namespace camera
{
  // DEBUG : remove these functions later

  static void soft_assert(const char* func, int& status)
  {
    if (status != PCO_NOERROR)
      std::cout << func << "() call failed : error " << status << std::endl;
    status = PCO_NOERROR;
  }

  // ! DEBUG

  CameraPCO::CameraPCO(
    const std::string& ini_filepath,
    WORD camera_type)
    : Camera(ini_filepath)
    , device_(nullptr)
    , camera_type_(camera_type)
    , buffers_events_()
    , buffers_()
  {
    /* Buffers events initialization. */
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

    /* Close buffers events. */
    for (unsigned int i = 0;
      i < buffers_events_.size();
      ++i)
    {
      CloseHandle(buffers_events_[i]);
    }

    for (unsigned int i = 0;
      i < buffers_.size();
      ++i)
    {
      delete[] buffers_[i];
      buffers_[i] = nullptr;
    }
  }

  void CameraPCO::init_camera()
  {
    int status = PCO_NOERROR;
    if (PCO_OpenCamera(&device_, 0) != PCO_NOERROR)
      throw CameraException(CameraException::NOT_CONNECTED);
    
    /* Ensure that the camera is not in recording state. */
    stop_acquisition();

    /* Camera type checking. */
    PCO_CameraType str_camera_type;
    str_camera_type.wSize = sizeof(PCO_CameraType);
    status |= PCO_GetCameraType(device_, &str_camera_type);
    if (str_camera_type.wCamType != camera_type_)
    {
      PCO_CloseCamera(device_);
      throw CameraException(CameraException::NOT_CONNECTED);
    }

    bind_params();
    
    if (status != PCO_NOERROR)
      throw CameraException(CameraException::NOT_INITIALIZED);
  }

  void CameraPCO::start_acquisition()
  {
    int status = PCO_NOERROR;

    /* Note : The SDK recommands the following setting order
    ** when potentially using ROI and/or binning :
    ** binning -> ROI -> arm camera -> getSizes -> allocate buffers
    */

    status |= PCO_ArmCamera(device_);
    soft_assert("PCO_ArmCamera", status);
    
    /* Retrieve frame resolution. */
    status |= get_sensor_sizes();
    soft_assert("PCO_GetSizes", status);
    std::cout << "Getting size : " << actual_res_x_ << "," << actual_res_y_ << std::endl;

    /* Buffer memory allocation. */
    status |= allocate_buffers();
    soft_assert("PCO_AllocateBuffers", status);

    /* Set recording state to [run] */
    status |= PCO_SetRecordingState(device_, PCO_RECSTATE_RUN);
    soft_assert("PCO_SetRecordingState", status);

    /* Add buffers into queue. */
    for (unsigned int i = 0;
      i < buffers_.size();
      ++i)
    {
      status |= PCO_AddBufferExtern(
        device_,
        buffers_events_[i],
        0, 0, 0, 0,
        buffers_[i],
        buffer_size_,
        &buffers_driver_status_[i]);
      if (status != PCO_NOERROR)
        std::cout << "PCO_AddBufferExtern " << i << " failed" << std::endl;
    }

    if (status != PCO_NOERROR)
      throw CameraException(CameraException::CANT_START_ACQUISITION);
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
    PCO_CloseCamera(device_);

    for (unsigned int i = 0;
      i < buffers_.size();
      ++i)
    {
      delete[] buffers_[i];
      buffers_[i] = nullptr;
    }
  }

  void* CameraPCO::get_frame()
  {
    DWORD event_status;
    if ((event_status = WaitForMultipleObjects(
      static_cast<DWORD>(buffers_events_.size()),
      buffers_events_._Elems,
      FALSE,
      FRAME_TIMEOUT)) < WAIT_ABANDONED_0)
    {
      DWORD buffer_index = event_status - WAIT_OBJECT_0;
      PCO_AddBufferExtern(
        device_,
        buffers_events_[buffer_index],
        0, 0, 0, 0,
        buffers_[buffer_index],
        buffer_size_,
        &buffers_driver_status_[buffer_index]);

      // DEBUG
      // Checking camera status regularly
      {
        DWORD warnings, errors, status;
        PCO_GetCameraHealthStatus(device_, &warnings, &errors, &status);
        if (warnings != PCO_NOERROR)
          std::cout << "[WARNING] : " << warnings << std::endl;
        if (errors != PCO_NOERROR)
          std::cout << "[ERRORS] : " << errors << std::endl;
      }
      // ! DEBUG

      return buffers_[buffer_index];
    }

    throw CameraException(CameraException::CANT_GET_FRAME);
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
    buffer_size_ = actual_res_x_ * actual_res_y_ * sizeof(WORD);
    int status = PCO_NOERROR;

    /* TODO: Error checking new operator -> try/catch. */
    try
    {
      for (unsigned int i = 0;
        i < buffers_.size();
        ++i)
      {
        buffers_[i] = new unsigned short[buffer_size_]();

        SHORT buffer_nbr = -1;

        status |= PCO_AllocateBuffer(
          device_,
          &buffer_nbr,
          buffer_size_,
          &buffers_[i],
          &buffers_events_[i]);

        if (status != PCO_NOERROR)
        soft_assert("PCO_AllocateBuffer", status);
        /*assert(buffer_nbr == static_cast<SHORT>(i));*/
        std::cout << "Buffer number : " << buffer_nbr << "against " << i << std::endl;
      }
    }
    catch (std::bad_alloc& ba)
    {
      // Erasing "unused variable ba" warning.
      ba;
      throw CameraException(CameraException::MEMORY_PROBLEM);
    }

    return status;
  }
}