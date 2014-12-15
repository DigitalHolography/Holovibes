#ifndef CAMERA_EXCEPTION_HH
# define CAMERA_EXCEPTION_HH

# include <exception>

namespace camera
{
  class CameraException : public std::exception
  {
  public:
    enum camera_error
    {
      NOT_CONNECTED,
      NOT_INITIALIZED,
      MEMORY_PROBLEM,
      CANT_START_ACQUISITION,
      CANT_STOP_ACQUISITION,
      CANT_GET_FRAME,
      CANT_SHUTDOWN,
      CANT_SET_CONFIG,
    };

    CameraException(camera_error code)
      : std::exception()
      , code_(code)
    {}
    
    virtual ~CameraException()
    {}

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
    const camera_error code_;
  };
}

#endif /* !CAMERA_EXCEPTION_HH */
