#ifndef ERROR_HANDLER_HH
# define ERROR_HANDLER_HH

# include <string>
# include <vector>
# include <iostream>
# include <ctime>

# include "errors_enum.hh"

# define MSGS_ARRAY_SIZE 256

namespace error
{
  /* This class is a Singleton
  ** it uses a static object because the class needs to be thread-safe
  */
  class ErrorHandler
  {
  public:
    static ErrorHandler& get_instance()
    {
      return instance_;
    }

    void send_error(const e_errors code) const;
    void send_error(
      const e_errors code,
      const std::string module_name) const;
    void send_error(const char* msg) const;
    void send_error(
      const char* msg,
      const std::string module_name) const;

  private:
    ErrorHandler()
    {
      load_msgs();
    }

    /* Copy is not allowed. */
    ErrorHandler(const ErrorHandler&)
    {}
    ErrorHandler& operator=(const ErrorHandler&)
    {}
    ~ErrorHandler()
    {}

    void load_msgs();
    std::string current_time() const;

  private:
    static ErrorHandler instance_;
    char* msgs_[MSGS_ARRAY_SIZE];
  };
}

#endif