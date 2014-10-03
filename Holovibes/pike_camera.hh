#ifndef PIKE_CAMERA_HH
# define PIKE_CAMERA_HH

# include <string>

/* Disable warnings for external header FGCamera.h. */
# pragma warning (push, 0)
# include <FGCamera.h>
# pragma warning (pop)

# include <iostream>
# include "camera.hh"

namespace camera
{
  class PikeCamera : public Camera
  {
  public:
    PikeCamera()
      : Camera()
    {
    }

    virtual ~PikeCamera()
    {
    }

    virtual bool init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;

    virtual void* get_frame() override;

  private:
    CFGCamera cam_;
    FGFRAME fgframe_;

    //Retrieve camera name (vendor and model from the device API)
    std::string get_name_from_device();
  };
}

#endif