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
 * Camera XIQ's. */
#pragma once

# include <Windows.h>
# include <xiApi.h>
# include <vector>

# include <camera.hh>

namespace camera
{
  class CameraXib : public Camera
  {
  public:
    CameraXib();

    virtual ~CameraXib()
    {
      /* Ensure that the camera is closed in case of exception. */
      shutdown_camera();
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

  private:
    HANDLE device_; //!< Pointer to the camera Xiq object provided by the API.

    XI_IMG frame_; //!< Buffer used for frame acquisition.

    float gain_; //!< Gain in dB.

	static const int real_width_ = 4096;
	static const int real_height_ = 3072;

    /*!
     * * 1: 1x1 sensor pixel  = 1 image pixel
     * * 2: 2x2 sensor pixels = 1 image pixel
     * * 4: 4x4 sensor pixels = 1 image pixel */
    unsigned int downsampling_rate_;

    //!< Downsampling method.
    /*! * XI_BINNING  0: pixels are interpolated - better image
      * * XI_SKIPPING 1 : pixels are skipped - higher frame rate */
    XI_DOWNSAMPLING_TYPE downsampling_type_;

    //!< RAW, 8/16-bit...
    /*!
     * * XI_MONO8
     * * XI_MONO16
     * * XI_RAW8
     * * XI_RAW16
     */
    XI_IMG_FORMAT img_format_;

    //!< How the camera should manage its buffer(s).
    /*!
     * * XI_BP_UNSAFE: User gets pointer to internally allocated circular
     * buffer and data may be overwritten by device.
     * * XI_BP_SAFE: Data from device will be copied to user allocated buffer
     * or xiApi allocated memory.
     */
    XI_BP buffer_policy_;

    //!< Activate a hardware/software trigger or not.
    /*!
    * * XI_TRG_OFF : Capture of next image is automatically started after previous.
    * * XI_TRG_EDGE_RISING: Capture is started on rising edge of selected input.
    * * XI_TRG_EDGE_FALLING: Capture is started on falling edge of selected input.
    * * XI_TRG_SOFTWARE: Capture is started with software trigger.
    */
    XI_TRG_SOURCE trigger_src_;

    int roi_x_; //!< ROI offset on X axis. Values start from 0.
    int roi_y_; //!< ROI offset on Y axis. Values start from 0.
    int roi_width_; //!< In pixels.
    int roi_height_; //!< In pixels.

	std::vector<char> buffer_rescale_;
  };
}