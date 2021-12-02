/*! \file
 *
 * Camera XIQ's. */
#pragma once

#include <Windows.h>
#include <xiApi.h>
#include <vector>
#include <iostream>

#include "camera.hh"
#include "camera_exception.hh"

namespace camera
{
class CameraXib : public Camera
{
  public:
    CameraXib();

    virtual ~CameraXib()
    {
        // Ensure that the camera is closed in case of exception.
        try
        {
            shutdown_camera();
        }
        // We can't throw in a destructor, but there's nothing to do on error
        catch (CameraException&)
        {
        }
    }

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual CapturedFramesDescriptor get_frames() override;

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

  private:
    /*! \brief Pointer to the camera Xiq object provided by the API. */
    HANDLE device_;
    /*! \brief Buffer used for frame acquisition. */
    XI_IMG frame_;
    /*! \brief Gain in dB. */
    float gain_;

    static const int real_width_ = 4704;
    static const int real_height_ = 3424;

    /*! \brief Downsampling rate
     *
     * * 1: 1x1 sensor pixel  = 1 image pixel
     * * 2: 2x2 sensor pixels = 1 image pixel
     * * 4: 4x4 sensor pixels = 1 image pixel
     */
    unsigned int downsampling_rate_;

    /*! \brief Downsampling method.
     * * XI_BINNING  0: pixels are interpolated - better image
     * * XI_SKIPPING 1 : pixels are skipped - higher frame rate
     */
    XI_DOWNSAMPLING_TYPE downsampling_type_;

    /*! \brief RAW, 8/16-bit...
     *
     * * XI_MONO8
     * * XI_MONO16
     * * XI_RAW8
     * * XI_RAW16
     */
    XI_IMG_FORMAT img_format_;

    /*! \brief How the camera should manage its buffer(s).
     *
     * * XI_BP_UNSAFE: User gets pointer to internally allocated circular
     * buffer and data may be overwritten by device.
     * * XI_BP_SAFE: Data from device will be copied to user allocated buffer
     * or xiApi allocated memory.
     */
    XI_BP buffer_policy_;

    /*! \brief Activate a hardware/software trigger or not.
     *
     * * XI_TRG_OFF : Capture of next image is automatically started after previous.
     * * XI_TRG_EDGE_RISING: Capture is started on rising edge of selected input.
     * * XI_TRG_EDGE_FALLING: Capture is started on falling edge of selected input.
     * * XI_TRG_SOFTWARE: Capture is started with software trigger.
     */
    XI_TRG_SOURCE trigger_src_;

    /*! \brief ROI offset on X axis. Values start from 0. */
    int roi_x_;
    /*! \brief ROI offset on Y axis. Values start from 0. */
    int roi_y_;
    /*! \brief In pixels. */
    int roi_width_;
    /*! \brief In pixels. */
    int roi_height_;

    std::vector<char> buffer_rescale_;
};
} // namespace camera
