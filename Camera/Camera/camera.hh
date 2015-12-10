#pragma once

# include <icamera.hh>
# include <frame_desc.hh>
# include <string>
# include <fstream>
# include <boost/property_tree/ini_parser.hpp>
# include <boost/property_tree/ptree.hpp>
# include <Windows.h>

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
    /*! Construct a blank camera object.
     * Try to open the corresponding configuration file, if any, and parse it with Boost
     * to extract some useful data for further configuration. */
    Camera(const std::string& ini_filepath)
      : desc_()
      , name_("Unknown")
      , exposure_time_(0.0f)
      , dll_instance_(nullptr)
      , create_logfile_(nullptr)
      , log_msg_(nullptr)
      , close_logfile_(nullptr)
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

    // Loading all utilities functions in the CamUtils DLL.
    void load_utils()
    {
      dll_instance_ = LoadLibraryW(L"CameraUtils.dll");
      if (!dll_instance_)
        throw std::runtime_error("Unable to load CameraUtils DLL.");

      create_logfile_ = reinterpret_cast<void(*)(std::string)>(GetProcAddress(dll_instance_, "create_logfile"));
      if (!create_logfile_)
        throw std::runtime_error("Unable to fetch create_log function.");
      log_msg_ = reinterpret_cast<void(*)(std::string)>(GetProcAddress(dll_instance_, "log_msg"));
      if (!log_msg_)
        throw std::runtime_error("Unable to fetch write_log function.");
      close_logfile_ = reinterpret_cast<void(*)()>(GetProcAddress(dll_instance_, "close_logfile"));
      if (!close_logfile_)
        throw std::runtime_error("Unable to fetch close_log function.");
      allocate_memory_ = reinterpret_cast<void(*)(void**, const std::size_t)>(GetProcAddress(dll_instance_, "allocate_memory"));
      if (!allocate_memory_)
        throw std::runtime_error("Unable to fetch allocate_memory function.");
      free_memory_ = reinterpret_cast<void(*)(void*)>(GetProcAddress(dll_instance_, "free_memory"));
      if (!free_memory_)
        throw std::runtime_error("Unable to free allocated memory.");
    }

  protected:
    FrameDescriptor desc_; //!< Frame descriptor updated by cameras.

    /* Theses parameters are common to all cameras.
     * Others parameters such as binning, gain ... belongs to
     * specific camera inherited class. */

    std::string name_;

    float exposure_time_;

    HINSTANCE dll_instance_; //!< Handle to the CamUtils DLL.

    void(*create_logfile_)(std::string);
    void(*log_msg_)(std::string);
    void(*close_logfile_)();
    void(*allocate_memory_)(void** buf, const std::size_t size);
    void(*free_memory_)(void* buf);

  private:
    std::string ini_path_; //!< INI configuration file's absolute path.

    std::ifstream  ini_file_; //!< INI configuration file data stream.

    boost::property_tree::ptree ini_pt_; //!< INI property tree, containing extracted data.
  };
}