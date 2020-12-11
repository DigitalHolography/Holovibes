/* **************************************************************************** */
/*                       ,,                     ,,  ,,                          */
/* `7MMF'  `7MMF'       `7MM       `7MMF'   `7MF'db *MM                         */
/*   MM      MM           MM         `MA     ,V      MM                         */
/*   MM      MM  ,pW"Wq.  MM  ,pW"Wq. VM:   ,V `7MM  MM,dMMb.   .gP"Ya  ,pP"Ybd */
/*   MMmmmmmmMM 6W'   `Wb MM 6W'   `Wb MM.  M'   MM  MM    `Mb ,M'   Yb 8I   `" */
/*   MM      MM 8M     M8 MM 8M     M8 `MM A'    MM  MM     M8 8M"""""" `YMMMa. */
/*   MM      MM YA.   ,A9 MM YA.   ,A9  :MM;     MM  MM.   ,M9 YM.    , L.   I8 */
/* .JMML.  .JMML.`Ybmd9'.JMML.`Ybmd9'    VF    .JMML.P^YbmdP'   `Mbmmd' M9mmmP' */
/*                                                                              */
/* **************************************************************************** */

/*! \file
 *
 * Common functionalities to all cameras.*/
#pragma once

# include <boost/property_tree/ini_parser.hpp>
# include <Windows.h>
# include <icamera.hh>
# include <frame_desc.hh>

namespace camera
{
  //! Adding to the ICamera interface datas and INI file loading.
  /*! Although each camera is different, a group of functionalities specific
   * to Holovibes are common to them, and defined here.
   *
   * Each camera is provided a INI file by Holovibes developers, which is a
   * really simple configuration format. The INI file contains options that
   * will be set automatically at camera startup; they are located in Holovibes/
   * main directory.
   * For more information on INI files, you can follow this link :
   * https://en.wikipedia.org/wiki/INI_file
   *
   * The Camera class provides methods to parse the INI file and keep data stored
   * for later use. */
  class Camera : public ICamera
  {
  public:
    virtual ~Camera()
    {
    }

    const FrameDescriptor& get_fd() const override
    {
      return fd_;
    }

    const float get_pixel_size() const override
    {
      return pixel_size_;
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
    /*! Construct a blank camera object.
     * Try to open the corresponding configuration file, if any, and parse it with Boost
     * to extract some useful data for further configuration. */
    Camera(const std::string& ini_filepath)
      : fd_()
      , name_("Unknown")
      , exposure_time_(0.0f)
	    , pixel_size_(0.0f)
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

    //! Get the Boost parser object that extracted data from the INI file.
    const boost::property_tree::ptree& get_ini_pt() const
    {
      if (!ini_file_is_open())
      {
        assert(!"Impossible case");
        throw std::exception("Attempt to get ini property tree when file is not opened");
      }
      return ini_pt_;
    }

    //! Load default parameters for the camera.
    /*! Fill default values in class fields and frame descriptor
     * (e.g. exposure time, ROI, fps). Each camera model has specific capabilities,
     * which is why further classes inherit from Camera to implement their behaviours
     * with appropriate their SDK.
     *
     * The camera must work with these default, fallback settings. */
    virtual void load_default_params() = 0;

    //! Load parameters from the INI file and store them (into private attributes).
    /*! Reads the file stream opened to the INI file, and fill the parser
     * object with corresponding data. */
    virtual void load_ini_params() = 0;

    /*! Set parameters with data loaded with load_ini_params().
     *
     * This method shall use the camera's API to properly modify its configuration.
     * Validity checking for any parameter is enclosed in this method. */
    virtual void bind_params() = 0;

  protected:
    FrameDescriptor fd_; //!< Frame descriptor updated by cameras.

    /* Theses parameters are common to all cameras.
     * Others parameters such as binning, gain ... belongs to
     * specific camera inherited class. */

    std::string name_;

	// Exposure time in �s
    float exposure_time_;
	  float pixel_size_;

	std::ifstream  ini_file_; //!< INI configuration file data stream.

  private:
    std::string ini_path_; //!< INI configuration file's absolute path.

    boost::property_tree::ptree ini_pt_; //!< INI property tree, containing extracted data.
  };
}