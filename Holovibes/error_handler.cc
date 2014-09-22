#include "error_handler.hh"

namespace error
{
  ErrorHandler ErrorHandler::instance_ = ErrorHandler::ErrorHandler();

  void ErrorHandler::load_errors_msgs()
  {
#define ERR_MSG(code, msg) errors_msgs[code] = msg;
    // Here all .def includes
#include "errors_pike_camera.def"
#undef ERR_MSG
  }
}