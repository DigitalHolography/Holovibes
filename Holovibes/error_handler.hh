#ifndef ERROR_HANDLER_HH
# define ERROR_HANDLER_HH

# include <string>

namespace error
{
  // This class is a Singleton
  class ErrorHandler
  {
  public:
    static ErrorHandler& get_instance()
    {
      if (instance_ == NULL)
        instance_ = new ErrorHandler();
      else
        return *instance_;
    }

    static void kill_instance()
    {
      if (instance_ != NULL)
        delete instance_;
    }

  private:
    ErrorHandler();
    ErrorHandler(const ErrorHandler&) {};
    ~ErrorHandler();
    ErrorHandler& operator=(const ErrorHandler&) {};

    static ErrorHandler* instance_;
  };
}

#endif