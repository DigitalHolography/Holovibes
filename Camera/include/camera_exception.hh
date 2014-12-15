#ifndef CAMERA_EXCEPTION_HH
# define CAMERA_EXCEPTION_HH

# include <exception>
# include <string>

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

    CameraException(std::string name, camera_error code)
      : std::exception()
      , name_(name)
      , code_(code)
    {}

    virtual const char* what() const override;

    const std::string& get_name() const
    {
      return name_;
    }

    const char* match_error() const;

  private:
    // Object is non copyable
    CameraException& operator=(const CameraException&) = delete;

    const std::string name_;
    const camera_error code_;
  };
}

#endif /* !CAMERA_EXCEPTION_HH */
