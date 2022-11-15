#pragma once

#include "internal_conversion.hh"

namespace camera
{

template <typename T>
Parallel_cvtColor<T>::Parallel_cvtColor(cv::Mat& src, cv::Mat& dst, float* coeffs)
    : m_src_(src)
    , m_dst_(dst)
    , coeffs_(coeffs)
{
}

template <typename T>
void Parallel_cvtColor<T>::operator()(const cv::Range& range) const
{
    for (int row = range.start; row < range.end; row++)
    {
        T* src_ptr = m_src_.ptr<T>(row);
        T* dst_ptr = m_dst_.ptr<T>(row);

        for (int col = 0; col < m_src_.cols; col++)
        {
            dst_ptr[col] = src_ptr[3 * col + 0] * coeffs_[0] + src_ptr[3 * col + 1] * coeffs_[1] +
                           src_ptr[3 * col + 2] * coeffs_[2];
        }
    }
}

} // namespace camera