/*! \file
 *
 * \brief OpenCV camera for the masse
 *
 */

#include "camera_opencv.hh"

#include <iostream>

#include "camera_logger.hh"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"

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
        Logger::camera()->error("Could not open opencv.ini config file");
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

double CameraOpenCV::get_and_check(int param, double value, std::string param_str)
{
    double tmp_value = capture_device_.get(param);
    if (tmp_value != value)
    {
        Logger::camera()->warn("Cannot set parameter {} to {}, value will be {}.", param_str, value, tmp_value);
    }
    return tmp_value;
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
    /*!
     * explanation:
     * (format & CV_MAT_DEPTH_MASK) gives the depth code internal to opencv
     *
     *  depth code | value | depth (byte)
     * ------------+-------+-------------
     *       CV_8U | 0     | 1
     *       CV_8S | 1     | 1
     *      CV_16U | 2     | 2
     *      CV_16S | 3     | 2
     *      CV_32S | 4     | 4
     *      CV_32F | 5     | 4
     *      CV_64F | 6     | 8
     * CV_USRTYPE1 | 7     | ERROR
     *
     */

    fd_.depth = ((0x8442211 >> ((format & CV_MAT_DEPTH_MASK) * 4)) & 15);
    if (fd_.depth == 0)
    {
        Logger::camera()->error("camera depth is unknown");
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    fps_ = get_and_check(cv::CAP_PROP_FPS, fps_, "opencv.fps");
    fd_.width = get_and_check(cv::CAP_PROP_FRAME_WIDTH, fd_.width, "opencv.width");
    fd_.height = get_and_check(cv::CAP_PROP_FRAME_HEIGHT, fd_.height, "opencv.height");
}

void CameraOpenCV::init_camera()
{
    deviceID_ = 0;        /* open default camera */
    apiID_ = cv::CAP_ANY; /* autodetect default API */
    capture_device_.open(deviceID_, apiID_);
    if (!capture_device_.isOpened())
    {
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
    /*
     * TODO: change how colors are converted to grey
     *
     * problem:
     * cvtColor(COLOR_BGR2GRAY) use some arbitrary values to make the conversion
     * from documentation (https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html):
     * - grey = 0.299*R + 0.587*G + 0.114*B
     *
     * idea by Michael:
     * - get the mean of all 3 colors (mean_B, mean_G, mean_R)
     * - grey = R/mean_R + G/mean_G + B/mean_B
     */
    cv::cvtColor(frame_, frame_, cv::COLOR_BGR2GRAY);
    return CapturedFramesDescriptor(frame_.data);
}
ICamera* new_camera_device() { return new CameraOpenCV(); }
} // namespace camera