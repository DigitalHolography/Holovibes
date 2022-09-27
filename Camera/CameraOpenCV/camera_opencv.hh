/*! \file
 *
 * \brief Default Camera using openCV VideoCapture to provide an "all-default" camera mode.
 */
#pragma once

#include "camera.hh"
#include "camera_exception.hh"
#include "opencv2/videoio.hpp"

namespace camera
{
class CameraOpenCV : public Camera
{
  public:
    CameraOpenCV();

    virtual ~CameraOpenCV() {}

    virtual void init_camera() override;
    virtual void start_acquisition() override;
    virtual void stop_acquisition() override;
    virtual void shutdown_camera() override;
    virtual CapturedFramesDescriptor get_frames() override;

  private:
    virtual void load_default_params() override;
    virtual void load_ini_params() override;
    virtual void bind_params() override;

    cv::Mat frame_;
    cv::VideoCapture capture_device_;
    int deviceID_;
    int apiID_;
    unsigned int fps_; 
};
} // namespace camera