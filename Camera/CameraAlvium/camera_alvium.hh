/*! \file
 *
 * Camera Alvium-1800-u/2050's. */
#pragma once

#include <Windows.h>
#include <VmbCPP/VmbCPP.h>

#include "camera.hh"

namespace camera
{
/*! \class CameraIds
 *
 * \brief
 */
class CameraAlvium : public Camera
{
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
};
} // namespace camera