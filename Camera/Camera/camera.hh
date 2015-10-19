#ifndef CAMERA_HH
# define CAMERA_HH

# include <icamera.hh>
# include <frame_desc.hh>

# include <string>
# include <fstream>
# include <boost/property_tree/ini_parser.hpp>
# include <boost/property_tree/ptree.hpp>
# include <Windows.h>

namespace camera
{
  /*! \class Camera
   *
   * \brief Add to ICamera interface datas and INI file loading.
   */
  class Camera : public ICamera
  {
  public:
    virtual ~Camera()
    {
    }

    const FrameDescriptor& get_frame_descriptor() const override
    {
      return desc_;
    }
    const char* get_name() const override
    {
      return name_.c_str();
    }
    const char* get_ini_path() const override
    {
      return ini_path_.c_str();
    }
  protected:
    /*! \brief Camera contructor
     *
     * Try to open the INI file, if any, parse it with Boost. */
    Camera(const std::string& ini_filepath)
      : desc_()
      , name_("Unknown")
      , exposure_time_(0.0f)
      , dll_instance_(nullptr)
      , create_log_(nullptr)
      , write_log_(nullptr)
      , ini_path_(ini_filepath)
      , ini_file_(ini_filepath, std::ifstream::in)
      , ini_pt_()
    {
      if (ini_file_is_open())
        boost::property_tree::ini_parser::read_ini(ini_file_, ini_pt_);
    }

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

    /*! \brief Load default parameters
     *
     * Fill default values in class fields and frame descriptor. It depends on
     * the camera model. It ensure the camera will work with these default
     * settings. */
    virtual void load_default_params() = 0;
    /*! Load parameters from INI file. */
    virtual void load_ini_params() = 0;
    /*! Send current parameters to camera API.
     *
     * Use once parameters has been loaded with load_default_params or
     * load_ini_params. Checks if parameters are valid. */
    virtual void bind_params() = 0;

    // Loading all utilities functions in the CamUtils DLL.
    void load_utils()
    {
      dll_instance_ = LoadLibrary("CameraUtils.dll");
      if (!dll_instance_)
        throw std::runtime_error("Unable to load CameraUtils DLL.");

      create_log_ = reinterpret_cast<FnUtil>(GetProcAddress(dll_instance_, "create_logfile"));
      if (!create_log_)
        throw std::runtime_error("Unable to fetch create_log functions.");
      write_log_ = reinterpret_cast<FnUtil>(GetProcAddress(dll_instance_, "log_msg"));
      if (!write_log_)
        throw std::runtime_error("Unable to fetch write_log functions.");
    }

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

    /* All CamUtils functions, and the DLL handle with it. */
    HINSTANCE dll_instance_;

    using FnUtil = void(*)(std::string);
    FnUtil create_log_;
    FnUtil write_log_;

  private:
    /*! INI configuration file path */
    std::string              ini_path_;
    /*! INI configuration file of camera. */
    std::ifstream            ini_file_;
    /*! INI property tree. */
    boost::property_tree::ptree ini_pt_;
  };
}
#endif /* !CAMERA_HH */