#pragma once

#include <concepts>
#include <array>

#include "opencv2/core.hpp"

namespace camera
{

template <typename T>
class Parallel_BGR_to_gray : public cv::ParallelLoopBody
{
  private:
    cv::Mat &m_src_bgr_, &m_dst_gray_;
    std::array<float, 3> coeffs_;

  public:
    Parallel_BGR_to_gray(cv::Mat& src, cv::Mat& dst, std::array<float, 3> coeffs);
    virtual void operator()(const cv::Range& range) const CV_OVERRIDE;
};

void internal_BGR_to_gray(cv::Mat& in, cv::Mat& out, std::array<float, 3> coeffs);
} // namespace camera

#include "internal_conversion.hxx"