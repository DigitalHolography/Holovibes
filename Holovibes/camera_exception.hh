#ifndef CAMERA_EXCEPTION_HH
# define CAMERA_EXCEPTION_HH

# include <exception>
# include <string>

namespace camera
{
  class CameraException : public std::exception
  {
  public:
    CameraException(std::string camera_name)
      : std::exception()
      , camera_name_(camera_name)
    {}

    virtual const char* what() const = 0;

    const std::string& get_camera_name() const
    {
      return camera_name_;
    }

  protected:
    const std::string camera_name_;

  private:
    // Object is non copyable
    CameraException& operator=(const CameraException&) = delete;
  };
}

#endif /* !CAMERA_EXCEPTION_HH */
