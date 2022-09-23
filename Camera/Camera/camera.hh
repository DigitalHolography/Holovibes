/*! \file
 *
 * \brief Common functionalities to all cameras.*/
#pragma once

#include <boost/property_tree/ini_parser.hpp>
#include <Windows.h>
#include "icamera.hh"
#include "frame_desc.hh"
#include "camera_config.hh"
#include "../../Holovibes/includes/holovibes_config.hh"
#include <spdlog/spdlog.h>

namespace camera
{
/*! \brief Adding to the ICamera interface datas and INI file loading.
 *
 * Although each camera is different, a group of functionalities specific
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
 * for later use.
 */

class Camera : public ICamera
{
  public:
    virtual ~Camera() {}

    const FrameDescriptor& get_fd() const override { return fd_; }

    const float get_pixel_size() const override { return pixel_size_; }

    const char* get_name() const override { return name_.c_str(); }

    const char* get_ini_name() const override { return ini_name_.c_str(); }

  protected:
    /*! \brief Construct a blank camera object.
     *
     * Try to open the corresponding configuration file, if any, and parse it
     * with Boost to extract some useful data for further configuration.
     */
    Camera(const std::string& ini_filename)
        : fd_()
        , name_("Unknown")
        , exposure_time_(0.0f)
        , pixel_size_(0.0f)
        , ini_pt_()
    {
        ini_name_ = (__CAMERAS_CONFIG_FOLDER_PATH__ / ini_filename).string();
        ini_file_ = std::ifstream(ini_name_);
        if (ini_file_is_open())
            boost::property_tree::ini_parser::read_ini(ini_file_, ini_pt_);
    }

    bool ini_file_is_open() const { return ini_file_.is_open(); }

    /*! \brief Get the Boost parser object that extracted data from the INI file. */
    const boost::property_tree::ptree& get_ini_pt() const
    {
        if (!ini_file_is_open())
        {
            assert(!"Impossible case");
            throw std::exception("Attempt to get ini property tree when file is not opened");
        }
        return ini_pt_;
    }

    /*! \brief Load default parameters for the camera.
     *
     * Fill default values in class fields and frame descriptor
     * (e.g. exposure time, ROI, fps). Each camera model has specific
     * capabilities, which is why further classes inherit from Camera to
     * implement their behaviours with appropriate their SDK.
     *
     * The camera must work with these default, fallback settings.
     */
    virtual void load_default_params() = 0;

    /*! \brief Load parameters from the INI file and store them (into private attributes).
     *
     * Reads the file stream opened to the INI file, and fill the parser object with corresponding data.
     */
    virtual void load_ini_params() = 0;

    /*! \brief Set parameters with data loaded with load_ini_params().
     *
     * This method shall use the camera's API to properly modify its
     * configuration. Validity checking for any parameter is enclosed in this
     * method.
     */
    virtual void bind_params() = 0;

  protected:
    /*! \brief Frame descriptor updated by camera */
    FrameDescriptor fd_;

    /* Theses parameters are common to all cameras.
     * Others parameters such as binning, gain ... belongs to
     * specific camera inherited class.
     */

    std::string name_;

    // #TODO Find the unit of measurement
    /*! \brief Exposure time in ?s */
    float exposure_time_;
    float pixel_size_;

    /*! \brief INI configuration file data stream */
    std::ifstream ini_file_;

  private:
    /*! \brief INI configuration file's absolute path */
    std::string ini_name_;

    /*! \brief INI property tree, containing extracted data. */
    boost::property_tree::ptree ini_pt_;
};
} // namespace camera
