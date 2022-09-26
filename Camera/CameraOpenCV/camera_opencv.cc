#include "camera_opencv.hh"

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <iostream>

namespace camera
{
CameraOpenCV::CameraOpenCV()
    : Camera("opencv.ini")
{
    name_ = "OpenCV";

    load_default_params();
    if (ini_file_is_open())
    {
        load_ini_params();
        ini_file_.close();
    }
    else
    {
        // FIXME LOG : Could not open opencv.ini config file
        throw CameraException(CameraException::NOT_INITIALIZED);
    }
    init_camera();
}

void CameraOpenCV::~CameraOpenCV() { return; }

void CameraOpenCV::load_default_params()
{
    fd_.depth = 2;
    fd_.byteEndian = Endianness::LittleEndian;
}

void CameraOpenCV::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    fd_.width = pt.get<unsigned short>("opencv.width", fd_.width);
    fd_.height = pt.get<unsigned short>("opencv.height", fd_.height);
    fps_ = pt.get<unsigned int>("opencv.fps", fps_);
}

void CameraOpenCV::bind_params()
{
    capture_device_.set(CAP_PROP_FPS, fps_);
    capture_device_.set(CAP_PROP_FRAME_WIDTH, fd_.width);
    capture_device_.set(CAP_PROP_FRAME_HEIGHT, fd_.height);
    capture_device_.set(CAP_PROP_FORMAT, -1);
}

void CameraOpenCV::init_camera()
{
    deviceID_ = 0;        // open default camera
    apiID_ = cv::CAP_ANY; // autodetect default API
    capture_device_.open(deviceID, apiID);
    if (!cpature_device_.isOpened())
    {
        // FIXME LOG : Could not connect the camera opencv
        throw CameraException(CameraException::NOT_CONNECTED);
    }
    bind_params();
}

void CameraOpenCV::start_acquisition() { return; }

void CameraOpenCV::stop_acquisition() { return; }

void CameraOpenCV::shutdown_camera() { capture_device_.release(); }

CapturedFramesDescriptor get_frames()
{
    frame_ = capture_device_.read();
    return new CapturedFramesDescriptor(frame_);
}
} // namespace camera