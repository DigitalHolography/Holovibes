#ifndef ICAMERA_HH
# define ICAMERA_HH

#ifdef CAMERA_EXPORTS
# define CAMERA_API __declspec(dllexport)
#else
# define CAMERA_API __declspec(dllimport)
#endif

namespace camera
{
  struct FrameDescriptor;

  static int FRAME_TIMEOUT = 10000; // FIXME

  /*! Abstract Camera class. */
  class ICamera
  {
  public:
    ICamera()
    {}
    virtual ~ICamera()
    {}

    virtual const FrameDescriptor& get_frame_descriptor() const = 0;
    virtual const char* get_name() const = 0;
    virtual const char* get_ini_path() const = 0;

    virtual void init_camera() = 0;
    virtual void start_acquisition() = 0;
    virtual void stop_acquisition() = 0;
    virtual void shutdown_camera() = 0;
    virtual void* get_frame() = 0;
  private:
    // Object is non copyable
    ICamera& operator=(const ICamera&) = delete;
    ICamera(const ICamera&) = delete;
  };

  extern "C"
  {
    CAMERA_API ICamera* new_camera_device();
  }
}

#endif /* !ICAMERA_HH */
