#pragma once
#include "camera.hh"
#include "camera_exception.hh"
#include <camera.h>

namespace camera
{
class CameraPCO_Edge5_5 : public Camera
{
  public:
    CameraPCO_Edge5_5();
    virtual ~CameraPCO_Edge5_5 ();
    
    pco::Image current_image_;
    pco::Camera pco_camera_; // PCO camera object
    // Camera parameters
    unsigned int roi_x_;        // ROI X start position (0-based)
    unsigned int roi_y_;        // ROI Y start position (0-based)
    unsigned int roi_width_;    // ROI width in pixels
    unsigned int roi_height_;   // ROI height in pixels
    double frame_period_;       // Time between frame starts (seconds)
    unsigned int trigger_mode_; // Trigger mode (0=auto, 1=external)
    unsigned int pixel_rate_;   // Pixel rate (0=default)
    std::string pixel_format_; // Pixel format (e.g., "Mono8", "Mono16")

    virtual void load_ini_params() override;
    virtual void shutdown_camera() override;
    virtual void load_default_params() override;
    virtual void bind_params() override;
    virtual const float get_pixel_size() const override;
    virtual const char* get_name() const override;
    virtual const char* get_ini_name() const override;
    virtual int get_temperature() const override;
    virtual int get_camera_fps() const override;
    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual CapturedFramesDescriptor get_frames() override;
    

};
} // namespace camera