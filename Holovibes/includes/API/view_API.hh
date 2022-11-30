#pragma once

#include "API_detail.hh"

namespace holovibes::api
{

inline ImageTypeEnum get_image_type() { return api::detail::get_value<ImageType>(); }
inline void set_image_type(ImageTypeEnum value) { return api::detail::set_value<ImageType>(value); }

inline bool get_fft_shift_enabled() { return api::detail::get_value<FftShiftEnabled>(); }
inline void set_fft_shift_enabled(bool value) { return api::detail::set_value<FftShiftEnabled>(value); }

inline bool get_cuts_view_enabled() { return api::detail::get_value<CutsViewEnable>(); }
inline void set_cuts_view_enabled(bool value) { api::detail::set_value<CutsViewEnable>(value); }

inline bool get_lens_view_enabled() { return api::detail::get_value<LensViewEnabled>(); }
inline void set_lens_view_enabled(bool value) { api::detail::set_value<LensViewEnabled>(value); }

inline bool get_chart_display_enabled() { return api::detail::get_value<ChartDisplayEnabled>(); }
inline void set_chart_display_enabled(bool value) { api::detail::set_value<ChartDisplayEnabled>(value); }

inline bool get_raw_view_enabled() { return api::detail::get_value<RawViewEnabled>(); }
inline void set_raw_view_enabled(bool value) { api::detail::set_value<RawViewEnabled>(value); }

inline bool get_filter2d_view_enabled() noexcept { return api::detail::get_value<Filter2DViewEnabled>(); }
inline void set_filter2d_view_enabled(bool value) noexcept
{
    return api::detail::set_value<Filter2DViewEnabled>(value);
}

inline const ViewAccuXY& get_view_accu_x() noexcept { return api::detail::get_value<ViewAccuX>(); }
inline const ViewAccuXY& get_view_accu_y() noexcept { return api::detail::get_value<ViewAccuY>(); }
inline const ViewAccuPQ& get_view_accu_p() noexcept { return api::detail::get_value<ViewAccuP>(); }
inline const ViewAccuPQ& get_view_accu_q() noexcept { return api::detail::get_value<ViewAccuQ>(); }
inline TriggerChangeValue<ViewAccuXY> change_view_accu_x() noexcept { return api::detail::change_value<ViewAccuX>(); }
inline TriggerChangeValue<ViewAccuXY> change_view_accu_y() noexcept { return api::detail::change_value<ViewAccuY>(); }
inline TriggerChangeValue<ViewAccuPQ> change_view_accu_p() noexcept { return api::detail::change_value<ViewAccuP>(); }
inline TriggerChangeValue<ViewAccuPQ> change_view_accu_q() noexcept { return api::detail::change_value<ViewAccuQ>(); }

inline const ViewXYZ& get_view_xy() noexcept { return api::detail::get_value<ViewXY>(); }
inline const ViewXYZ& get_view_xz() noexcept { return api::detail::get_value<ViewXZ>(); }
inline const ViewXYZ& get_view_yz() noexcept { return api::detail::get_value<ViewYZ>(); }
inline TriggerChangeValue<ViewXYZ> change_view_xy() noexcept { return api::detail::change_value<ViewXY>(); }
inline TriggerChangeValue<ViewXYZ> change_view_xz() noexcept { return api::detail::change_value<ViewXZ>(); }
inline TriggerChangeValue<ViewXYZ> change_view_yz() noexcept { return api::detail::change_value<ViewYZ>(); }

inline ViewWindow get_view_filter2d() { return api::detail::get_value<ViewFilter2D>(); }
inline TriggerChangeValue<ViewWindow> change_view_filter2d() { return api::detail::change_value<ViewFilter2D>(); }

inline const ReticleStruct& get_reticle() { return api::detail::get_value<Reticle>(); }
inline TriggerChangeValue<ReticleStruct> change_reticle() { return api::detail::change_value<Reticle>(); }

inline bool get_renorm_enabled() { return api::detail::get_value<RenormEnabled>(); }
inline void set_renorm_enabled(bool value) { api::detail::set_value<RenormEnabled>(value); }

} // namespace holovibes::api
