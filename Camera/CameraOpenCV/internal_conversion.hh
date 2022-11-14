#pragma once

#include <concepts>

#include "opencv2/core.hpp"

namespace camera
{
template <int type>
concept mat_type = type == CV_8UC1 || type == CV_16UC1;

class Parallel_cvtColor_default : public cv::ParallelLoopBody
{
  protected:
    cv::Mat &m_src_, &m_dst_;
    float* coeffs_;

  public:
    Parallel_cvtColor_default(cv::Mat& src, cv::Mat& dst, float* coeffs);
};

template <int type>
requires mat_type<type>
class Parallel_cvtColor : public Parallel_cvtColor_default
{
  public:
    Parallel_cvtColor(cv::Mat& src, cv::Mat& dst, float* coeffs);
};

void internal_convertColor(cv::Mat& in, cv::Mat& out, float* coeffs);
} // namespace camera

#include "internal_conversion.hxx"