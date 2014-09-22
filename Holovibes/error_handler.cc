#include "error_handler.hh"

namespace error
{
  ErrorHandler ErrorHandler::instance_ = ErrorHandler::ErrorHandler();

  void ErrorHandler::load_errors_msgs()
  {
    errors_msgs_ = std::vector<std::string>(256, "");

#define ERR_MSG(code, msg) errors_msgs_[code] = msg;
    // Here all .def includes
#undef ERR_MSG
  }

  bool ErrorHandler::send_error(e_errors code)
  {
    std::cerr << "Error: " << errors_msgs_[code] << std::endl;
    return true;
  }

  bool ErrorHandler::send_error(e_errors code, std::string module_name)
  {
    std::cerr << "Error: Module: "
      << module_name << ": "
      << errors_msgs_[code] << std::endl;
    return true;
  }
}