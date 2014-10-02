#ifndef EXCEPTION_CAMERA_HH
# define EXCEPTION_CAMERA_HH

# include <exception>
# include <string>

namespace camera
{
  class ExceptionCamera : public std::exception
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

    ExceptionCamera(std::string name, camera_error code)
      : std::exception()
      , name_(name)
      , code_(code)
    {}

    virtual const char* what() const override;

    const std::string& get_name() const
    {
      return name_;
    }

    const std::string match_error() const;

  private:
    // Object is non copyable
    ExceptionCamera& operator=(const ExceptionCamera&) = delete;

    const std::string name_;
    const camera_error code_;
  };
}

#endif /* !EXCEPTION_CAMERA_HH */
