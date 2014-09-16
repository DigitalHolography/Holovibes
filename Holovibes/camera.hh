#ifndef CAMERA_HH
# define CAMERA_HH

# include <string>

namespace cam_driver
{
  class Camera
  {
  public:
    enum endianness
    {
      BIG_ENDIAN,
      LITTLE_ENDIAN
    };

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

    // Setters

    // Virtual methods
    virtual bool init_camera() = 0;
    virtual void start_acquisition() = 0;
    virtual void stop_acquisition() = 0;
    virtual void shutdown_camera() = 0;

  private:
    std::string name_;
    enum endianness endianness_;

    // Buffer properties
    bool external_buffer_support_;
    bool non_paged_buffer_support_;

    int frame_height_;
    int frame_width_;
    int frame_depth_;
  };
}

#endif /* !CAMERA_HH */