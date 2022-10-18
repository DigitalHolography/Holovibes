#pragma once

#include "API_detail.hh"

namespace holovibes::api
{
inline ImgType get_img_type() { return api::detail::get_value<ImgTypeParam>(); }

inline bool get_fft_shift_enabled() { return api::detail::get_value<FftShiftEnabled>(); }

inline bool get_cuts_view_enabled() { return api::detail::get_value<CutsViewEnabled>(); }
inline void set_cuts_view_enabled(bool value) { api::detail::set_value<CutsViewEnabled>(value); }

inline bool get_lens_view_enabled() { return api::detail::get_value<LensViewEnabled>(); }
inline void set_lens_view_enabled(bool value) { api::detail::set_value<LensViewEnabled>(value); }

inline bool get_chart_display_enabled() { return api::detail::get_value<ChartDisplayEnabled>(); }
inline bool get_chart_record_enabled() { return api::detail::get_value<ChartRecordEnabled>(); }

inline bool get_raw_view_enabled() { return api::detail::get_value<RawViewEnabled>(); }

inline bool get_reticle_display_enabled() { return api::detail::get_value<ReticleDisplayEnabled>(); }
inline void set_reticle_display_enabled(bool value) { api::detail::set_value<ReticleDisplayEnabled>(value); }

inline View_XY get_view_x(void) { return api::detail::get_value<ViewX>(); }
inline View_XY get_view_y(void) { return api::detail::get_value<ViewY>(); }

inline bool get_filter2d_view_enabled() noexcept { return api::detail::get_value<Filter2DViewEnabled>(); }

inline uint get_accu_x() noexcept { return api::detail::get_value<ViewAccuX>(); }
inline uint get_accu_y() noexcept { return api::detail::get_value<ViewAccuY>(); }
inline uint get_accu_p() noexcept { return api::detail::get_value<ViewAccuP>(); }
inline uint get_accu_q() noexcept { return api::detail::get_value<ViewAccuQ>(); }

inline WindowKind get_current_window_type() noexcept { return api::detail::get_value<CurrentWindowKind>(); }

} // namespace holovibes::api
