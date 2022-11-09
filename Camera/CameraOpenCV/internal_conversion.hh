#pragma once

#include "opencv2/core.hpp"

namespace camera
{
class Parallel_cvtColor_default : public cv::ParallelLoopBody
{
  protected:
    cv::Mat &m_src_, &m_dst_;
    float* coeffs_;

  public:
    Parallel_cvtColor_default(cv::Mat& src, cv::Mat& dst, float* coeffs);
};

template <int mat_type>
class Parallel_cvtColor : public Parallel_cvtColor_default
{
  public:
    Parallel_cvtColor(cv::Mat& src, cv::Mat& dst, float* coeffs) = 0;
};

void internal_convertColor(cv::Mat& in, cv::Mat& out, float* coeffs);
} // namespace camera

#include "internal_conversion.hxx"