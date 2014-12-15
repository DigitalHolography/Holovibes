#ifndef CAMERA_PIMPL_HH
# define CAMERA_PIMPL_HH

# include <camera.hh>
# include <frame_desc.hh>

# include <string>
# include <fstream>
# include <boost/property_tree/ini_parser.hpp>
# include <boost/property_tree/ptree.hpp>

namespace camera
{
  class Camera::CameraPimpl
  {
  public:
    CameraPimpl(const char* const ini_filepath)
      : desc_()
      , name_("Unknown")
      , exposure_time_(0.0f)
      , ini_path_(ini_filepath)
      , ini_file_(ini_filepath, std::ifstream::in)
      , ini_pt_()
    {
      if (ini_file_is_open())
        boost::property_tree::ini_parser::read_ini(ini_file_, ini_pt_);
    }

    ~CameraPimpl() = default;

    const std::string& get_ini_path() const
    {
      return ini_path_;
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

  public:
    /*! Frame descriptor updated by cameras. */
    FrameDescriptor          desc_;

    /* Theses parameters are common to all cameras.
     * Others parameters such as binning, gain ... belongs to
     * specific camera inherited class. */
    /*! Name of the camera. */
    std::string              name_;
    /*! Exposure time of the camera. */
    float                    exposure_time_;

  private:
    /*! INI configuration file path */
    std::string              ini_path_;
    /*! INI configuration file of camera. */
    std::ifstream            ini_file_;
    /*! INI property tree. */
    boost::property_tree::ptree ini_pt_;
  };
}

#endif /* !CAMERA_PIMPL_HH */