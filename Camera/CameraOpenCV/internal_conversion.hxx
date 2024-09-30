#pragma once

#include "internal_conversion.hh"

namespace camera
{

template <typename T>
Parallel_BGR_to_gray<T>::Parallel_BGR_to_gray(cv::Mat& src, cv::Mat& dst, std::array<float, 3> coeffs)
    : m_src_bgr_(src)
    , m_dst_gray_(dst)
    , coeffs_(coeffs)
{
}

template <typename T>
void Parallel_BGR_to_gray<T>::operator()(const cv::Range& range) const
{
    for (int row = range.start; row < range.end; row++)
    {
        T* bgr_ptr = m_src_bgr_.ptr<T>(row);
        T* gray_ptr = m_dst_gray_.ptr<T>(row);

        for (int col = 0; col < m_src_bgr_.cols; col++)
        {
            gray_ptr[col] = bgr_ptr[3 * col + 0] * coeffs_[0] + bgr_ptr[3 * col + 1] * coeffs_[1] +
                            bgr_ptr[3 * col + 2] * coeffs_[2];
        }
    }
}

} // namespace camera