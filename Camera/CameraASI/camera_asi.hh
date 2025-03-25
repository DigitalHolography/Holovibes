#pragma once

#include <ASICamera2.h>        // ASI SDK header provided by ZWOptical
#include "camera.hh"           // Base Camera interface
#include "camera_exception.hh" // Exception class for camera errors

namespace camera
{

class CameraAsi : public Camera
{
  public:
    CameraAsi();
    virtual ~CameraAsi();

    // Initialize the camera (open + init)
    virtual void init_camera() override;

    // Start video acquisition (capture)
    virtual void start_acquisition() override;

    // Stop video acquisition
    virtual void stop_acquisition() override;

    // Shutdown and release camera resources
    virtual void shutdown_camera() override;

    // Retrieve a captured frame (descriptor must be defined in camera.hh)
    virtual CapturedFramesDescriptor get_frames() override;

  private:
    // Load configuration parameters from an ini file (if applicable)
    virtual void load_ini_params() override;

    // Load default parameters for the camera
    virtual void load_default_params() override;

    // Bind the parameters to internal variables if needed
    virtual void bind_params() override;

    int cameraID;            // Unique camera ID (default: 0)
    ASI_CAMERA_INFO camInfo; // Camera information structure
    bool isInitialized;      // Flag to indicate if the camera is initialized
    int resolution_width_;
    int resolution_height_;
    int pixel_depth_value_;
    int gain_value_;
    int exposure_time_;
};

} // namespace camera