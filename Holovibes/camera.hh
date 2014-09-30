#ifndef CAMERA_HH
# define CAMERA_HH

# include "frame_desc.hh"

# include <string>

namespace camera
{
  class Camera
  {
  public:
    // Default constants
    static const int FRAME_TIMEOUT = 1000;

    Camera()
    {}

    ~Camera()
    {}

    // Getters
    const std::string& get_name() const
    {
      return name_;
    }
    const s_frame_desc& get_frame_descriptor() const
    {
      return desc_;
    }
    float get_pixel_size() const
    {
      return pixel_size_;
    }

    // Virtual methods
    virtual bool init_camera() = 0;
    virtual void start_acquisition() = 0;
    virtual void stop_acquisition() = 0;
    virtual void shutdown_camera() = 0;

    virtual void* get_frame() = 0;

    /* Protected contains fields that are mutables for inherited class. */
  protected:
    s_frame_desc             desc_;
    double                   exposure_time_;
    unsigned short           frame_rate_;
    /*! Name of the camera. */
    std::string              name_;
    /*! Size of a pixel in micrometer. */
    float                    pixel_size_;
  private:
    // Object is non copyable
    Camera& operator=(const Camera&) = delete;
    Camera(const Camera&) = delete;
  };
}

#endif /* !CAMERA_HH */