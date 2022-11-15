#include <chrono>

#include "camera_exception.hh"
#include "camera_logger.hh"
#include "camera_opencv.hh"
#include "internal_conversion.hh"

namespace camera
{

template <typename T>
void templated_internal_convertColor(cv::Mat& in, cv::Mat& out, float* coeffs, int mat_type)
{

    cv::Mat dst(in.rows, in.cols, mat_type);

    Parallel_cvtColor<T> parallel_cvtColor(in, dst, coeffs);
    cv::parallel_for_(cv::Range(0, in.rows), parallel_cvtColor);

    out = dst;
}

void internal_convertColor(cv::Mat& in, cv::Mat& out, float* coeffs)
{
    switch (in.depth())
    {
    case CV_8U:
        templated_internal_convertColor<uchar>(in, out, coeffs, CV_8UC1);
        break;
    case CV_16U:
        templated_internal_convertColor<ushort>(in, out, coeffs, CV_16UC1);
        break;

    default:
        Logger::camera()->error("camera depth is not supported");
        throw CameraException(CameraException::CANT_GET_FRAME);
        break;
    }
}
} // namespace camera