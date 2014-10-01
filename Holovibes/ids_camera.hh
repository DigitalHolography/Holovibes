#ifndef IDS_CAMERA_HH
# define IDS_CAMERA_HH

# include <uEye.h>
# include "camera.hh"
# include "camera_exception.hh"

namespace camera
{
  class IDSCamera : public Camera
  {
  public:
    IDSCamera()
      : Camera()
    {
      desc_.width = 2048;
      desc_.height = 2048;
      desc_.bit_depth = 8;
    }

    ~IDSCamera()
    {
    }

    virtual bool init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;

    virtual void* get_frame() override;

  private:
    HIDS cam_;
    char* frame_;
    int frame_mem_pid_;
  };
}

#endif /* !IDS_CAMERA_HH */