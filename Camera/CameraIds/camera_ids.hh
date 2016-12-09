/*! \file
 *
 * Camera IDS */
#pragma once

# include <camera.hh>

/* Disable warnings. */
# pragma warning (push, 0)
# include <uEye.h>
# pragma warning (pop)

namespace camera
{
  //!< IDS camera.
  class CameraIds : public Camera
  {
  public:
    CameraIds()
      : Camera("ids.ini")
    {
      name_ = "IDS CMOSIS CMV 4000";

      load_default_params();
      if (ini_file_is_open())
        load_ini_params();

	  if (ini_file_is_open())
		  ini_file_.close();
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
    /*! Format gain, it should be between 0 and 100 as it is a coefficient.
    * \return 0 if gain < 0 or gain > 100; else returns gain. */
    int format_gain() const;

    /*! Retrieve subsampling mode code from a string.
    * \return The corresponding API-defined code, or the subsampling-disabling code
    * if the value is invalid. */
    int get_subsampling_mode(const std::string ui) const;

    /*! Retrieve binning mode code from user input string.
    * \return The corresponding API-defined code, or the binning-disabling code
    * if the value is invalid. */
    int get_binning_mode(const std::string ui) const;

    /*! Retrieve color mode code from user input string.
    * \return The corresponding API-defined code, or the raw 8-bit format
    * if the value is invalid. */
    int get_color_mode(const std::string ui);

    /*! Retrieve trigger mode code from user input string.
    * \return The corresponding API-defined code, or the trigger-disabling code
    * if the value is invalid. */
    int get_trigger_mode(const std::string ui) const;

  private:
    HIDS cam_; //!< camera handler

    char* frame_; //!< frame pointer

    int frame_mem_pid_; //!< frame associated memory

    unsigned int gain_; //!< Gain

    int subsampling_; //!< Subsampling value (2x2, 4x4 ...)

    int binning_; //!< Binning value (2x2, 4x4 ...)

    int color_mode_; //!< Image format (also called color mode)

    int aoi_x_; //!< Area Of Interest (AOI) x

    int aoi_y_; //!< Area Of Interest (AOI) y

    int aoi_width_; //!< Area Of Interest (AOI) width

    int aoi_height_; //!< Area Of Interest (AOI) height

    int trigger_mode_; //!< Trigger mode
  };
}