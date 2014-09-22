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
    std::cerr << current_time()
      << "Error: " << errors_msgs_[code] << std::endl;
    return true;
  }

  bool ErrorHandler::send_error(e_errors code, std::string module_name)
  {
    std::cerr << current_time()
      << "Error: Module: "
      << module_name << ": "
      << errors_msgs_[code] << std::endl;
    return true;
  }

  bool ErrorHandler::send_error(char* msg)
  {
    std::cerr << current_time()
      << "Error: " << msg << std::endl;
    return true;
  }

  bool ErrorHandler::send_error(char* msg, std::string module_name)
  {
    std::cerr << current_time()
      << "Error: Module: "
      << module_name << ": "
      << msg << std::endl;
    return true;
  }

  std::string ErrorHandler::current_time()
  {
    time_t rawtime = time(0);
    struct tm now;
    localtime_s(&now, &rawtime);

    const size_t size = 26;
    char buffer[size];
    asctime_s(buffer, size, &now);

    return std::string(buffer);
  }
}