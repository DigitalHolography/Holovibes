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
 * Camera Ixon */
#pragma once

# include <camera.hh>

namespace camera
{
  //!< Camera Ixon Zyla.
  class CameraIxon : public Camera
  {
  public:
    CameraIxon();
    virtual ~CameraIxon();

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual void* get_frame() override;

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

  private:
    int vertical_shift_speed_;
    int horizontal_shift_speed_;

    int gain_mode_;

    float kinetic_time_;

    unsigned short r_x;
    unsigned short r_y;

    unsigned short *output_image_;

    long device_handle;

    unsigned short* image_;

    int trigger_mode_;

    int shutter_close_;
    int shutter_open_;

    int ttl_;

    int shutter_mode_;
    /* FIXME: What acquisiton means ? */
    int acquisiton_mode_;
    int read_mode_;
  };
}