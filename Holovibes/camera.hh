#ifndef CAMERA_HH
# define CAMERA_HH

# include <string>
# include "endianness.hh"
# include "frame.hh"

namespace camera
{
  class Camera
  {
  public:
    Camera(std::string name)
      : name_(name)
    {}

    ~Camera()
    {}

    // Getters
    const std::string& get_name() const
    {
      return name_;
    }

    enum endianness get_endianness() const
    {
      return endianness_;
    }

    // Virtual methods
    virtual bool init_camera() = 0;
    virtual void start_acquisition() = 0;
    virtual void stop_acquisition() = 0;
    virtual void shutdown_camera() = 0;

    virtual Frame& get_frame() = 0;

  protected:
    std::string name_;

  private:
    enum endianness endianness_;
    Frame frame_;
  };
}

#endif /* !CAMERA_HH */