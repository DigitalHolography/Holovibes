#pragma once

#include "API_detail.hh"

namespace holovibes::api
{
inline ImgType get_img_type() { return api::detail::get_value<ImgTypeParam>(); }
inline void set_img_type(ImgType _img_type) { return api::detail::set_value<ImgTypeParam>(_img_type); }

inline bool get_fft_shift_enabled() { return api::detail::get_value<FftShiftEnabled>(); }
inline void set_fft_shift_enabled(bool value) { return api::detail::set_value<FftShiftEnabled>(value); }

inline bool get_filter2d_enabled() { return api::detail::get_value<Filter2DEnabled>(); }

inline bool get_cuts_view_enabled() { return api::detail::get_value<CutsViewEnabled>(); }
inline void set_cuts_view_enabled(bool value) { api::detail::set_value<CutsViewEnabled>(value); }

inline bool get_lens_view_enabled() { return api::detail::get_value<LensViewEnabled>(); }
inline void set_lens_view_enabled(bool value) { api::detail::set_value<LensViewEnabled>(value); }

inline bool get_chart_display_enabled() { return api::detail::get_value<ChartDisplayEnabled>(); }
inline bool get_chart_record_enabled() { return api::detail::get_value<ChartRecordEnabled>(); }

inline bool get_raw_view_enabled() { return api::detail::get_value<RawViewEnabled>(); }

inline bool get_reticle_display_enabled() { return api::detail::get_value<ReticleDisplayEnabled>(); }
inline void set_reticle_display_enabled(bool value) { api::detail::set_value<ReticleDisplayEnabled>(value); }

inline bool get_filter2d_view_enabled() noexcept { return api::detail::get_value<Filter2DViewEnabled>(); }
inline void set_filter2d_view_enabled(bool value) noexcept
{
    return api::detail::set_value<Filter2DViewEnabled>(value);
}

inline const View_XY& get_view_accu_x() noexcept { return api::detail::get_value<ViewAccuX>(); }
inline const View_XY& get_view_accu_y() noexcept { return api::detail::get_value<ViewAccuY>(); }
inline const View_PQ& get_view_accu_p() noexcept { return api::detail::get_value<ViewAccuP>(); }
inline const View_PQ& get_view_accu_q() noexcept { return api::detail::get_value<ViewAccuQ>(); }
inline TriggerChangeValue<View_XY> change_view_accu_x() noexcept { return api::detail::change_value<ViewAccuX>(); }
inline TriggerChangeValue<View_XY> change_view_accu_y() noexcept { return api::detail::change_value<ViewAccuY>(); }
inline TriggerChangeValue<View_PQ> change_view_accu_p() noexcept { return api::detail::change_value<ViewAccuP>(); }
inline TriggerChangeValue<View_PQ> change_view_accu_q() noexcept { return api::detail::change_value<ViewAccuQ>(); }

inline const View_XYZ& get_view_xy() noexcept { return api::detail::get_value<ViewXY>(); }
inline const View_XYZ& get_view_xz() noexcept { return api::detail::get_value<ViewXZ>(); }
inline const View_XYZ& get_view_yz() noexcept { return api::detail::get_value<ViewYZ>(); }
inline TriggerChangeValue<View_XYZ> change_view_xy() noexcept { return api::detail::change_value<ViewXY>(); }
inline TriggerChangeValue<View_XYZ> change_view_xz() noexcept { return api::detail::change_value<ViewXZ>(); }
inline TriggerChangeValue<View_XYZ> change_view_yz() noexcept { return api::detail::change_value<ViewYZ>(); }

inline View_Window get_filter2d() { return api::detail::get_value<Filter2D>(); }
inline TriggerChangeValue<View_Window> change_filter2d() { return api::detail::change_value<Filter2D>(); }

inline float get_reticle_scale() { return api::detail::get_value<ReticleScale>(); }
inline void set_reticle_scale(float value) { api::detail::set_value<ReticleScale>(value); }

void display_reticle(bool value);
void reticle_scale(float value);

void create_holo_window(ushort window_size);
// TODO: param index is imposed by MainWindow behavior, and should be replaced by something more generic like
// dictionary
void refresh_view_mode(ushort window_size, uint index);
void set_view_mode(const std::string& value, std::function<void()> callback);

void set_filter2d_view(bool checked, uint auxiliary_window_max_size);
void set_filter2d(bool checked);

inline bool get_renorm_enabled() { return GSH::instance().get_value<RenormEnabled>(); }
inline void set_renorm_enabled(bool value) { GSH::instance().set_value<RenormEnabled>(value); }

} // namespace holovibes::api
