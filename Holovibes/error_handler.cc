#include "error_handler.hh"

#define TIME_STR_SIZE 26
#define ERROR_FORMAT "[%s][ERROR] %s\n"
#define ERROR_MOD_FORMAT "[%s][ERROR][%s] %s\n"

namespace holovibes
{
  ErrorHandler ErrorHandler::instance_ = ErrorHandler::ErrorHandler();

  void ErrorHandler::load_msgs()
  {
#define ERR_MSG(Code, Msg) msgs_[Code] = (Msg);
    // Here all .def includes
#undef ERR_MSG
  }

  void ErrorHandler::send_error(const e_errors code) const
  {
    fprintf(stderr, ERROR_FORMAT,
      current_time().c_str(),
      msgs_[code]);
  }

  void ErrorHandler::send_error(
    const e_errors code,
    const std::string module_name) const
  {
    fprintf(stderr, ERROR_MOD_FORMAT,
      current_time().c_str(),
      module_name.c_str(),
      msgs_[code]);
  }

  void ErrorHandler::send_error(const char* msg) const
  {
    fprintf(stderr, ERROR_FORMAT,
      current_time(),
      msg);
  }

  void ErrorHandler::send_error(
    const char* msg,
    const std::string module_name) const
  {
    fprintf(stderr, ERROR_MOD_FORMAT,
      current_time().c_str(),
      module_name.c_str(),
      msg);
  }

  std::string ErrorHandler::current_time() const
  {
    time_t rawtime = time(0);
    struct tm now;
    localtime_s(&now, &rawtime);

    char buffer[TIME_STR_SIZE];
    asctime_s(buffer, TIME_STR_SIZE, &now);
    /* Remove the '\n' character. */
    buffer[24] = '\0';

    return buffer;
  }
}
