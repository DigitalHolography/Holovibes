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

#define MAX_HEIGHT 512
#define MAX_WIDTH 512

namespace camera
{
/*! \class CameraIds
 *
 * \brief
 */
class CameraAlvium : public Camera
{
    class FrameObserver : public VmbCPP::IFrameObserver
    {
      public:
        FrameObserver(VmbCPP::CameraPtr Camera_ptr, VmbInt64_t size_frame, CameraAlvium& CameraAlvium);
        void FrameReceived(const VmbCPP::FramePtr pFrame);

      private:
        VmbInt64_t size_frame_;
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

    VmbCPP::VmbSystem& api_vmb_;
    VmbCPP::CameraPtr camera_ptr_;
    VmbCPP::FramePtrVector frames_; // Frame array FIXME but ok
    std::queue<unsigned char*> waiting_queue_;
    VmbFeaturePersistSettings_t settingsStruct_;

    int width_;
    int height_;
};
} // namespace camera