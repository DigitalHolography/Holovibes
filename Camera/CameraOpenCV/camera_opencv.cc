/*! \file
 *
 * \brief OpenCV camera for the masse
 *
 */

#include "camera_opencv.hh"

#include <iostream>
#include <chrono>

#include "camera_logger.hh"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"
#include "internal_conversion.hh"

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
    pixel_size_ = pt.get<float>("opencv.pixel_size", pixel_size_);

    std::string str = pt.get<std::string>("grayscale.method");
    if (str == "OPENCV")
        method_ = OPENCV;
    else if (str == "MANUAL")
    {
        method_ = MANUAL;
        grayscale_coeffs_[0] = pt.get<float>("grayscale.blue");
        grayscale_coeffs_[1] = pt.get<float>("grayscale.green");
        grayscale_coeffs_[2] = pt.get<float>("grayscale.red");
    }
    else if (str == "AUTO")
        method_ = AUTO;
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
}

void CameraOpenCV::init_camera()
{
    deviceID_ = 0;          /* open default camera */
    apiID_ = cv::CAP_DSHOW; /* autodetect default API */
    capture_device_.open(deviceID_, apiID_);
    if (!capture_device_.isOpened())
    {
        throw CameraException(CameraException::NOT_CONNECTED);
    }

    bind_params();

    capture_device_.read(frame_);

    int format = frame_.depth();
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

    fd_.depth = static_cast<PixelDepth>((0x8442211 >> ((format & CV_MAT_DEPTH_MASK) * 4)) & 0xf);
    if (fd_.depth == camera::PixelDepth::Bits0)
    {
        Logger::camera()->error("camera depth is unknown");
        throw CameraException(CameraException::NOT_INITIALIZED);
    }

    fps_ = get_and_check(cv::CAP_PROP_FPS, fps_, "opencv.fps");
    fd_.width = get_and_check(cv::CAP_PROP_FRAME_WIDTH, fd_.width, "opencv.width");
    fd_.height = get_and_check(cv::CAP_PROP_FRAME_HEIGHT, fd_.height, "opencv.height");

    if (method_ == AUTO)
    {
        cv::Scalar means = cv::mean(frame_);
        grayscale_coeffs_[0] = 1 / means[0];
        grayscale_coeffs_[1] = 1 / means[1];
        grayscale_coeffs_[2] = 1 / means[2];
        float sum = grayscale_coeffs_[0] + grayscale_coeffs_[1] + grayscale_coeffs_[2];
        grayscale_coeffs_[0] /= sum;
        grayscale_coeffs_[1] /= sum;
        grayscale_coeffs_[2] /= sum;
    }
}

void CameraOpenCV::start_acquisition() { return; }

void CameraOpenCV::stop_acquisition() { return; }

void CameraOpenCV::shutdown_camera() { capture_device_.release(); }

CapturedFramesDescriptor CameraOpenCV::get_frames()
{
    capture_device_.read(frame_);
    switch (method_)
    {
    case AUTO:
    case MANUAL:
        internal_BGR_to_gray(frame_, frame_, grayscale_coeffs_);
        break;
    case OPENCV:
        cv::cvtColor(frame_, frame_, cv::COLOR_BGR2GRAY);
        break;
    }
    return CapturedFramesDescriptor(frame_.data);
}

ICamera* new_camera_device() { return new CameraOpenCV(); }
} // namespace camera
