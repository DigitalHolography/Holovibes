#ifndef CAMERA_HH
# define CAMERA_HH

# include <string>
# include <fstream>

# include "frame_desc.hh"

namespace camera
{
  /*! Abstract Camera class. */
  class Camera
  {
  public:
    // Default constants
    static const int FRAME_TIMEOUT = 1000;

    /*! \brief Base class constructor. It opens the ini file if any,
    ** otherwise it will loads defaults parameters.
    ** \param ini_filepath INI camera configuration file path.
    */
    Camera(std::string& ini_filepath)
      : desc_()
      , name_("Unknown")
      , exposure_time_(0.0)
      , frame_rate_(0)
      , ini_file_(ini_filepath, std::ofstream::in)
    {}

    virtual ~Camera()
    {
      ini_file_.close();
    }

#pragma region getters
    const s_frame_desc& get_frame_descriptor() const
    {
      return desc_;
    }
    const std::string& get_name() const
    {
      return name_;
    }
    double get_exposure_time() const
    {
      return exposure_time_;
    }
    unsigned short get_frame_rate() const
    {
      return frame_rate_;
    }
#pragma endregion

    virtual bool init_camera() = 0;
    virtual void start_acquisition() = 0;
    virtual void stop_acquisition() = 0;
    virtual void shutdown_camera() = 0;
    virtual void* get_frame() = 0;

    /* protected methods */
  protected:
    virtual void load_default_params() = 0;
    virtual void load_ini_params() = 0;

    /* protected fields */
  protected:
    /*! Frame descriptor updated by cameras. */
    s_frame_desc             desc_;

    /*! Name of the camera. */
    std::string              name_;
    /*! Exposure time of the camera. */
    double                   exposure_time_;
    /*! Number of frames per second. */
    unsigned short           frame_rate_;

    /*! INI configuration file of camera. */
    std::ifstream            ini_file_;
  private:
    // Object is non copyable
    Camera& operator=(const Camera&) = delete;
    Camera(const Camera&) = delete;
  };
}

#endif /* !CAMERA_HH */
