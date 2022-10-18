#pragma once

#include "API_detail.hh"
#include "contrast_API.hh"

namespace holovibes::api
{
inline int get_filter2d_n1() { return api::detail::get_value<Filter2DN1>(); }
inline void set_filter2d_n1(int value)
{
    api::detail::set_value<Filter2DN1>(value);
    api::set_auto_contrast_all();
}

inline int get_filter2d_n2() { return api::detail::get_value<Filter2DN2>(); }
inline void set_filter2d_n2(int value)
{
    api::detail::set_value<Filter2DN2>(value);
    api::set_auto_contrast_all();
}

inline int get_filter2d_smooth_low() { return api::detail::get_value<Filter2DSmoothLow>(); }
inline void set_filter2d_smooth_low(int value) { api::detail::set_value<Filter2DSmoothLow>(value); }

inline int get_filter2d_smooth_high() { return api::detail::get_value<Filter2DSmoothHigh>(); }
inline void set_filter2d_smooth_high(int value) { api::detail::set_value<Filter2DSmoothHigh>(value); }

} // namespace holovibes::api
