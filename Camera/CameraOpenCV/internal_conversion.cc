#include <chrono>

#include "camera_exception.hh"
#include "camera_logger.hh"
#include "camera_opencv.hh"
#include "internal_conversion.hh"

namespace camera
{

Parallel_cvtColor_default::Parallel_cvtColor_default(cv::Mat& src, cv::Mat& dst, float* coeffs)
    : m_src_(src)
    , m_dst_(dst)
    , coeffs_(coeffs)
{
}

template <int mat_type>
void templated_internal_convertColor(cv::Mat& in, cv::Mat& out, float* coeffs)
{

    cv::Mat dst(in.rows, in.cols, mat_type);

    Parallel_cvtColor<mat_type> parallel_cvtColor(in, dst, coeffs);
    cv::parallel_for_(cv::Range(0, in.rows), parallel_cvtColor);

    out = dst;
}

void internal_convertColor(cv::Mat& in, cv::Mat& out, float* coeffs)
{
    cv::Mat dst;
    switch (in.depth())
    {
    case CV_8U:
        templated_internal_convertColor<CV_8UC1>(in, out, coeffs);
        break;
    case CV_16U:
        templated_internal_convertColor<CV_16UC1>(in, out, coeffs);
        break;

    default:
        Logger::camera()->error("camera depth is not supported");
        throw CameraException(CameraException::CANT_GET_FRAME);
        break;
    }
}
} // namespace camera