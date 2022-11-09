#pragma once

#include "internal_conversion.hh"

namespace camera
{

template <>
class Parallel_cvtColor<CV_8UC1> : public Parallel_cvtColor_default
{
  private:
    uchar *src_ptr_, *dst_ptr_;

  public:
    Parallel_cvtColor(cv::Mat& src, cv::Mat& dst, float* coeffs)
        : Parallel_cvtColor_default(src, dst, coeffs)
    {
        src_ptr_ = m_src_.ptr<uchar>();
        dst_ptr_ = m_dst_.ptr<uchar>();
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        for (int r = range.start; r < range.end; r++)
        {
            dst_ptr_[r] =
                src_ptr_[3 * r] * coeffs_[0] + src_ptr_[3 * r + 1] * coeffs_[1] + src_ptr_[3 * r + 2] * coeffs_[2];
        }
    }
};

template <>
class Parallel_cvtColor<CV_16UC1> : public Parallel_cvtColor_default
{
  private:
    ushort *src_ptr_, *dst_ptr_;

  public:
    Parallel_cvtColor(cv::Mat& src, cv::Mat& dst, float* coeffs)
        : Parallel_cvtColor_default(src, dst, coeffs)
    {
        src_ptr_ = m_src_.ptr<ushort>();
        dst_ptr_ = m_dst_.ptr<ushort>();
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        for (int r = range.start; r < range.end; r++)
        {
            dst_ptr_[r] =
                src_ptr_[3 * r] * coeffs_[0] + src_ptr_[3 * r + 1] * coeffs_[1] + src_ptr_[3 * r + 2] * coeffs_[2];
        }
    }
};
} // namespace camera