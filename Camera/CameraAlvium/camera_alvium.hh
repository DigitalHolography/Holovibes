/*! \file
 *
 * Camera Alvium-1800-u/2050's. */
#pragma once

#include <Windows.h>
#include <VmbCPP/VmbCPP.h>
#include <VmbC/VmbCommonTypes.h>
#include <vector>
#include <queue>

#include "camera.hh"

namespace camera
{
/*! \class CameraAlvium
 *
 * \brief
 *  Class for the camera Alvium
 */
class CameraAlvium : public Camera
{
    /*!
     *  \class FrameObserver
     *
     *  \brief FrameObserver is an observer who implement VmbCPP::IFrameObserver interface.
     *  The observer is used to get the frame from the camera when they arrive and give them to
     *  Holovibes when getFrame is called.
     */
    class FrameObserver : public VmbCPP::IFrameObserver
    {
      public:
        FrameObserver(VmbCPP::CameraPtr Camera_ptr, CameraAlvium& CameraAlvium);

        /*!
         * \brief Receive a frame and add it to the camera waiting_queue_.
         */
        void FrameReceived(const VmbCPP::FramePtr pFrame);

      private:
        /*! \brief The camera object to get the waiting_queue. */
        CameraAlvium& camera_alvium_;
    };

  public:
    CameraAlvium();

    virtual ~CameraAlvium()
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

    /*! \brief The API to interact with the camera. */
    VmbCPP::VmbSystem& api_vmb_;

    /*! \brief To store a pointer to the camera object. */
    VmbCPP::CameraPtr camera_ptr_;

    /*! \brief A temporary queue to store the frames when they are received by the observer and give them when getFrames
     * is called
     */
    std::queue<unsigned char*> waiting_queue_;

    /*! \brief The width of the frames taken by the camera. */
    int width_;

    /*! \brief The height of the frames taken by the camera. */
    int height_;

    /*! \brief Some params given by the ini file needed by the camera */

    std::string pixel_format_;
    bool reverse_x_;
    bool reverse_y_;
    double gamma_;
    double gain_;
    double lens_shading_value_;
    std::string intensity_auto_precedence_;
    std::string exposure_active_mode_;
    std::string exposure_auto_;
    double exposure_time_;
    std::string correction_mode_;
    std::string correction_selector_;
    VmbInt64_t contrast_bright_limit_;
    VmbInt64_t contrast_dark_limit_;
    bool contrast_enable_;
    VmbInt64_t contrast_shape_;
    double black_level_;
    VmbInt64_t binning_horizontal_;
    std::string binning_horizontal_mode_;
    VmbInt64_t binning_vertical_;
    std::string binning_vertical_mode_;
    double adaptive_noise_suppression_factor_;

    /*! \brief The max width of a frame. */
    static constexpr unsigned short MAX_WIDTH = 5496;

    /*! \brief The max height of a frame. */
    static constexpr unsigned short MAX_HEIGHT = 3672;
};
} // namespace camera
