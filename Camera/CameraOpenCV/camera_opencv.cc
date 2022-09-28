#include "camera_opencv.hh"

#include <iostream>

#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/imgproc.hpp"

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

void CameraOpenCV::load_default_params() { fd_.byteEndian = Endianness::LittleEndian; }

void CameraOpenCV::load_ini_params()
{
    const boost::property_tree::ptree& pt = get_ini_pt();

    fd_.width = pt.get<unsigned short>("opencv.width", fd_.width);
    fd_.height = pt.get<unsigned short>("opencv.height", fd_.height);
    fps_ = pt.get<unsigned int>("opencv.fps", fps_);
    pixel_size_ = pt.get<unsigned int>("opencv.pixel_size", pixel_size_);
}

void CameraOpenCV::bind_params()
{
    capture_device_.set(cv::CAP_PROP_FPS, fps_);
    capture_device_.set(cv::CAP_PROP_FRAME_WIDTH, fd_.width);
    capture_device_.set(cv::CAP_PROP_FRAME_HEIGHT, fd_.height);

    int format = capture_device_.get(cv::CAP_PROP_FORMAT);
    if (format == -1)
    {
        capture_device_.read(frame_);
        format = frame_.depth();
    }
    fd_.depth = ((0x8442211 >> ((format & CV_MAT_DEPTH_MASK) * 4)) & 15);

    int tmp_fps, tmp_width, tmp_height;
    if ((tmp_fps = capture_device_.get(cv::CAP_PROP_FPS)) != fps_)
    {
        // FIXME LOG:
        fps_ = tmp_fps;
    }
    if ((tmp_width = capture_device_.get(cv::CAP_PROP_FRAME_WIDTH)) != fd_.width)
    {
        // FIXME: LOG
        fd_.width = tmp_width;
    }
    if ((tmp_height = capture_device_.get(cv::CAP_PROP_FRAME_HEIGHT)) != fd_.height)
    {
        // FIXME: LOG
        fd_.height = tmp_height;
    }
}

void CameraOpenCV::init_camera()
{
    deviceID_ = 0;        // open default camera
    apiID_ = cv::CAP_ANY; // autodetect default API
    capture_device_.open(deviceID_, apiID_);
    if (!capture_device_.isOpened())
    {
        // FIXME LOG : Could not connect the camera opencv
        throw CameraException(CameraException::NOT_CONNECTED);
    }
    bind_params();
}

void CameraOpenCV::start_acquisition() { return; }

void CameraOpenCV::stop_acquisition() { return; }

void CameraOpenCV::shutdown_camera() { capture_device_.release(); }

CapturedFramesDescriptor CameraOpenCV::get_frames()
{
    capture_device_.read(frame_);
    cv::cvtColor(frame_, frame_, cv::COLOR_BGR2GRAY);
    return CapturedFramesDescriptor(frame_.data);
}
ICamera* new_camera_device() { return new CameraOpenCV(); }
} // namespace camera