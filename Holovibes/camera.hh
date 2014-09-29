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

    Camera(std::string name, s_roi *roi_info)
      : name_(name), roi_info_(roi_info)
    {

    }

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

    s_roi* get_roi()
    {
      return roi_info_;
    }

    // Virtual methods
    virtual bool init_camera() = 0;
    virtual void start_acquisition() = 0;
    virtual void stop_acquisition() = 0;
    virtual void shutdown_camera() = 0;

    virtual void* get_frame() = 0;

  protected:
    std::string name_;

  private:
    enum endianness endianness_;

    s_roi* roi_info_;
    int frame_height_;
    int frame_width_;
    unsigned char frame_bit_depth_;
  };

  typedef struct roi
  {
    int x_start;
    int y_start;
    int x_end;
    int y_end;
  }s_roi;
}

#endif /* !CAMERA_HH */