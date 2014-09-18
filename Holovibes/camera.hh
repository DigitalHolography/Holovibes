#ifndef CAMERA_HH
# define CAMERA_HH

# include <string>

namespace camera
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

    int get_frame_width() const
    {
      return frame_width_;
    }

    int get_frame_height() const
    {
      return frame_height_;
    }

    // Virtual methods
    virtual bool init_camera() = 0;
    virtual void start_acquisition() = 0;
    virtual void stop_acquisition() = 0;
    virtual void shutdown_camera() = 0;

  protected:
    std::string name_;

  private:
    enum endianness endianness_;

    int frame_height_;
    int frame_width_;
    unsigned char frame_bit_depth_;
  };
}

#endif /* !CAMERA_HH */