/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file camera_exception.hh
 *
 * Various set of exceptions related to the camera's operation.
 */
#pragma once

# include <exception>

namespace camera
{
  /*! A variety of exceptions for errors happening at different
   * stages of a camera's operation. */
  class CameraException : public std::exception
  {
  public:
    //!< The type of error encountered by the camera.
    enum camera_error
    {
      NOT_CONNECTED, //!< Camera needs to be powered on.
      NOT_INITIALIZED, //!< Startup failed.
      MEMORY_PROBLEM, //!< Buffer allocation / deallocation.
      CANT_START_ACQUISITION, //!< Acquisition setup failed.
      CANT_STOP_ACQUISITION, //!< Acquisition halting failed.
      CANT_GET_FRAME, //!< Current configuration is unusable or a frame was simply missed.
      CANT_SHUTDOWN, //!< Camera cannot power off.
      CANT_SET_CONFIG, //!< Some given configuration option is invalid.
    };

    /* \brief Copy constructor. */
    CameraException(const camera_error code)
      : code_(code)
    {
    }

    /*! Although we may need the copy constructor in order to pass to a function
    * an exception to be thrown, assignation does not make sense.
    */
    CameraException& operator=(const CameraException&) = delete;

    /* Destruct the exception. */
    virtual ~CameraException()
    {
    }

    /* \brief Return a string corresponding to the enum value. */
    virtual const char* what() const override
    {
      switch (code_)
      {
      case NOT_CONNECTED:
        return "is not connected";
      case NOT_INITIALIZED:
        return "could not be initialized.";
      case MEMORY_PROBLEM:
        return "memory troubles, can not access, "
          "allocate or bind camera memory.";
      case CANT_START_ACQUISITION:
        return "can't start acquisition.";
      case CANT_STOP_ACQUISITION:
        return "can't stop acquisition.";
      case CANT_GET_FRAME:
        return "can't get frame.";
      case CANT_SHUTDOWN:
        return "can't shut down camera.";
      case CANT_SET_CONFIG:
        return "can't set the camera configuration";
      default:
        return "unknown error";
      }
    }

  private:
    /* \brief Return code of the camera (enum). */
    const camera_error code_;
  };
}