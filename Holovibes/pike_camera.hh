#ifndef PIKE_CAMERA_HH
# define PIKE_CAMERA_HH

# include <string>
# include <FGCamera.h>
# include <iostream>
# include "camera.hh"

namespace camera
{
  class PikeCamera : public Camera
  {
  public:
    PikeCamera(std::string name)
      : Camera(name)
    {
    }

    ~PikeCamera()
    {
    }

    bool init_camera() override;
    void start_acquisition() override;
    void stop_acquisition() override;
    void shutdown_camera() override;

  private:
    CFGCamera cam_;

    //Retrieve camera name (vendor and model from the device API)
    std::string get_name_from_device();
  };
}

#endif