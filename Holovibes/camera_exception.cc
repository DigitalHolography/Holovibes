#include "stdafx.h"
#include "camera_exception.hh"

namespace camera
{
  const char* CameraException::what() const
  {
    std::string res = name_ + " " + match_error();
    return res.c_str();
  }

  const std::string CameraException::match_error() const
  {
    switch (code_)
    {
    case NOT_INITIALIZED:
      return "couldn't be initialized.";
    case CANT_START_ACQUISITION:
      return "can't start acquisition.";
    case CANT_STOP_ACQUISITION:
      return "can't stop acquisition.";
    case CANT_GET_FRAME:
      return "can't get frame.";
    case CANT_SHUTDOWN:
      return "can't shut down camera.";
    default:
      return "unknown error";
    }
  }
}