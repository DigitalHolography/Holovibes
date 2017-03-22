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
 * Camera Pike */
#pragma once

# include <iostream>
# include <string>

# include <camera.hh>

/* Disable warning. */
# pragma warning (push, 0)
# include <FGCamera.h>
# pragma warning (pop)

namespace camera
{
  //!< Pike camera.
  class CameraPike : public Camera
  {
  public:
    CameraPike()
      : Camera("pike.ini")
    {
      name_ = "Pike Kodak KAI 4022 F-421";

      load_default_params();
      if (ini_file_is_open())
        load_ini_params();

	  if (ini_file_is_open())
		  ini_file_.close();
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
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

    //!< Retrieve camera name (vendor and model from the device API)
    std::string get_name_from_device();

    unsigned long to_dcam_format() const;

    //!< Convert user input to speed parameter
    unsigned long to_speed() const;

  private:
    CFGCamera cam_;
    FGFRAME fgframe_;

    unsigned int subsampling_;
    unsigned long gain_;
    unsigned long brightness_;
    unsigned long gamma_;
    unsigned long speed_;

    unsigned long trigger_on_;
    unsigned long trigger_pol_;
    unsigned long trigger_mode_;

    int roi_startx_;
    int roi_starty_;
    int roi_width_;
    int roi_height_;
  };
}