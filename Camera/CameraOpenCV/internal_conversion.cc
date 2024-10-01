#include <chrono>

#include "camera_exception.hh"
#include "camera_logger.hh"
#include "camera_opencv.hh"
#include "internal_conversion.hh"

namespace camera
{

template <typename T>
void templated_internal_BGR_to_gray(cv::Mat& in, cv::Mat& out, std::array<float, 3> coeffs, int mat_type)
{

    cv::Mat dst(in.rows, in.cols, mat_type);

    Parallel_BGR_to_gray<T> parallel_BGR_to_gray(in, dst, coeffs);
    cv::parallel_for_(cv::Range(0, in.rows), parallel_BGR_to_gray);

    out = dst;
}

void internal_BGR_to_gray(cv::Mat& in, cv::Mat& out, std::array<float, 3> coeffs)
{
    switch (in.depth())
    {
    case CV_8U:
        templated_internal_BGR_to_gray<uchar>(in, out, coeffs, CV_8UC1);
        break;
    case CV_16U:
        templated_internal_BGR_to_gray<ushort>(in, out, coeffs, CV_16UC1);
        break;

    default:
        Logger::camera()->error("camera depth is not supported");
        throw CameraException(CameraException::CANT_GET_FRAME);
        break;
    }
}
} // namespace camera
