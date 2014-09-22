#ifndef ERROR_HANDLER_HH
# define ERROR_HANDLER_HH

# include <string>
# include <vector>
# include <iostream>
# include "errors_enum.hh"

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

    bool send_error(int error_code);

  private:
    static ErrorHandler instance_;
    std::vector<std::string> errors_msgs;

    ErrorHandler()
    {
      load_errors_msgs();
    }

    ErrorHandler(const ErrorHandler&)
    {
    }

    ~ErrorHandler()
    {
    }

    ErrorHandler& operator=(const ErrorHandler&)
    {
    }

    void load_errors_msgs();
  };
}

#endif