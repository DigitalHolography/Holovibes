#ifndef CAMERA_HH
# define CAMERA_HH

# include "frame_desc.hh"

# include <string>
# include <fstream>
# include <boost/property_tree/ini_parser.hpp>
# include <boost/property_tree/ptree.hpp>

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
    Camera(const std::string ini_filepath)
      : desc_()
      , name_("Unknown")
      , exposure_time_(0.0f)
      , ini_file_(ini_filepath, std::ofstream::in)
    {
      if (ini_file_is_open())
        boost::property_tree::ini_parser::read_ini(ini_file_, ini_pt_);
    }

    virtual ~Camera()
    {
      ini_file_.close();
    }

#pragma region getters
    const FrameDescriptor& get_frame_descriptor() const
    {
      return desc_;
    }
    const std::string& get_name() const
    {
      return name_;
    }
    float get_exposure_time() const
    {
      return exposure_time_;
    }
#pragma endregion

    virtual void init_camera() = 0;
    virtual void start_acquisition() = 0;
    virtual void stop_acquisition() = 0;
    virtual void shutdown_camera() = 0;
    virtual void* get_frame() = 0;

    /* protected methods */
  protected:
    bool ini_file_is_open() const
    {
      return ini_file_.is_open();
    }
    const boost::property_tree::ptree& get_ini_pt() const
    {
      if (!ini_file_is_open())
      {
        assert(!"Impossible case");
        throw std::exception("Attempt to get ini property tree when file is not opened");
      }
      return ini_pt_;
    }

    /* protected fields */
  protected:
    /*! Frame descriptor updated by cameras. */
    FrameDescriptor          desc_;

    /* Theses parameters are common to all cameras.
     * Others parameters such as binning, gain ... belongs to
     * specific camera inherited class. */
    /*! Name of the camera. */
    std::string              name_;
    /*! Exposure time of the camera. */
    float                    exposure_time_;

    /* private methods */
  private:
    /*! Load default parameters. */
    virtual void load_default_params() = 0;
    /*! Load parameters from INI file. */
    virtual void load_ini_params() = 0;
    /*! Send current parameters to camera API. */
    virtual void bind_params() = 0;

    /* private fields */
  private:
    /*! INI configuration file of camera. */
    std::ifstream            ini_file_;
    /*! INI property tree. */
    boost::property_tree::ptree ini_pt_;

    // Object is non copyable
    Camera& operator=(const Camera&) = delete;
    Camera(const Camera&) = delete;
  };
}

#endif /* !CAMERA_HH */
