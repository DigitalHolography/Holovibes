#ifndef CAMERA_IDS_HH
# define CAMERA_IDS_HH

# include <uEye.h>
# include "camera.hh"
# include "exception_camera.hh"

namespace camera
{
  class CameraIds : public Camera
  {
  public:
    CameraIds()
      : Camera("ids.ini")
    {
      load_default_params();
    }

    virtual ~CameraIds()
    {
    }

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;

    virtual void* get_frame() override;

    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

  private:
    /*! camera handler */
    HIDS cam_;

    /*! frame pointer */
    char* frame_;

    /*! frame associated memory */
    int frame_mem_pid_;
  };
}

#endif /* !CAMERA_IDS_HH */
