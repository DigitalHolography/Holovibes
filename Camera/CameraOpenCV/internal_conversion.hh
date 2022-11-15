#pragma once

#include <concepts>

#include "opencv2/core.hpp"

namespace camera
{

template <typename T>
class Parallel_cvtColor : public cv::ParallelLoopBody
{
  private:
    cv::Mat &m_src_, &m_dst_;
    float* coeffs_;

  public:
    Parallel_cvtColor(cv::Mat& src, cv::Mat& dst, float* coeffs);
    virtual void operator()(const cv::Range& range) const CV_OVERRIDE;
};

void internal_convertColor(cv::Mat& in, cv::Mat& out, float* coeffs);
} // namespace camera

#include "internal_conversion.hxx"