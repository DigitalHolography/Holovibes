#pragma once

#include "internal_conversion.hh"

namespace camera
{

template <>
class Parallel_cvtColor<CV_8UC1> : public Parallel_cvtColor_default
{
  public:
    Parallel_cvtColor(cv::Mat& src, cv::Mat& dst, float* coeffs)
        : Parallel_cvtColor_default(src, dst, coeffs)
    {
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        for (int row = range.start; row < range.end; row++)
        {
            uchar* src_ptr = m_src_.ptr<uchar>(row);
            uchar* dst_ptr = m_dst_.ptr<uchar>(row);

            for (int col = 0; col < m_src_.cols; col++)
            {
                dst_ptr[col] = src_ptr[3 * col + 0] * coeffs_[0] + src_ptr[3 * col + 1] * coeffs_[1] +
                               src_ptr[3 * col + 2] * coeffs_[2];
            }
        }
    }
};

template <>
class Parallel_cvtColor<CV_16UC1> : public Parallel_cvtColor_default
{
  public:
    Parallel_cvtColor(cv::Mat& src, cv::Mat& dst, float* coeffs)
        : Parallel_cvtColor_default(src, dst, coeffs)
    {
    }

    virtual void operator()(const cv::Range& range) const CV_OVERRIDE
    {
        for (int row = range.start; row < range.end; row++)
        {
            ushort* src_ptr = m_src_.ptr<ushort>(row);
            ushort* dst_ptr = m_dst_.ptr<ushort>(row);

            for (int col = 0; col < m_src_.cols; col++)
            {
                dst_ptr[col] = src_ptr[3 * col + 0] * coeffs_[0] + src_ptr[3 * col + 1] * coeffs_[1] +
                               src_ptr[3 * col + 2] * coeffs_[2];
            }
        }
    }
};
} // namespace camera