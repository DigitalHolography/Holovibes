#ifndef CAMERA_IDS_HH
# define CAMERA_IDS_HH

# include "camera.hh"

/* Disable warnings. */
# pragma warning (push, 0)
# include <uEye.h>
# pragma warning (pop)

namespace camera
{
  class CameraIds : public Camera
  {
  public:
    CameraIds()
      : Camera("ids.ini")
    {
      name_ = "ids";
      load_default_params();
      if (ini_file_is_open())
        load_ini_params();
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

    /*! Gain */
    unsigned int gain_;

    /*! Subsampling value (2x2, 4x4 ...) */
    int subsampling_;

    /*! Binning value (2x2, 4x4 ...) */
    int binning_;

    /*! Image format (also called color mode) */
    int color_mode_;

    /*! Area Of Interest (AOI) x */
    int aoi_x_;

    /*! Area Of Interest (AOI) y */
    int aoi_y_;

    /*! Area Of Interest (AOI) width */
    int aoi_width_;

    /*! Area Of Interest (AOI) height */
    int aoi_height_;

    /*! Trigger mode */
    int trigger_mode_;

  private:
    /*! Format gain, it should be between 0 and 100 as it is a coefficient */
    int format_gain();

    /*! Retreive subsampling mode code from user input string */
    int get_subsampling_mode(std::string ui);

    /*! Retreive binning mode code from user input string */
    int get_binning_mode(std::string ui);

    /*! Retreive color mode code from user input string */
    int get_color_mode(std::string ui);

    /*! Retreive trigger mode code from user input string */
    int get_trigger_mode(std::string ui);
  };
}

#endif /* !CAMERA_IDS_HH */
