#ifndef CAMERA_PIKE_HH
# define CAMERA_PIKE_HH

# include "camera.hh"
# include "camera_exception.hh"

/* Disable warning. */
# pragma warning (push, 0)
# include <FGCamera.h>
# pragma warning (pop)
# include <iostream>
# include <string>

namespace camera
{
  class CameraPike : public Camera
  {
  public:
    CameraPike()
      : Camera("pike.ini")
    {
      load_default_params();
      if (ini_file_is_open())
        load_ini_params();
    }

    virtual ~CameraPike()
    {
    }

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;

    virtual void* get_frame() override;

  private:
    CFGCamera cam_;
    FGFRAME fgframe_;

    unsigned int subsampling_;
    unsigned long gain_;
    unsigned long brightness_;
    unsigned long gamma_;

    int roi_startx_;
    int roi_starty_;
    int roi_width_;
    int roi_height_;
  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

    //Retrieve camera name (vendor and model from the device API)
    std::string get_name_from_device();

    unsigned long CameraPike::to_dcam_format();
  };
}

#endif /* !CAMERA_PIKE_HH */
