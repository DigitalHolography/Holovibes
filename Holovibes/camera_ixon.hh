#ifndef CAMERA_IXON_HH
# define CAMERA_IXON_HH

# include "camera.hh"
# include <atmcd32d.h>
# include <iostream>

namespace camera
{
  class CameraIxon : public Camera
  {
  public:
    CameraIxon();
    ~CameraIxon();

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual void* get_frame() override;

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

  private:
    long device_handle;
    unsigned short* image_;
    int trigger_mode_;
    float shutter_close_;
    float shutter_open_;
    int ttl_;
    int shutter_mode_;
    int acquisiton_mode_;
    int read_mode_;
  };
}
#endif /* !CAMERA_ZYLA_HH */