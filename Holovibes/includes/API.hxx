#pragma once

#include "API.hh"
#include "enum_record_mode.hh"

namespace holovibes::api
{
inline Computation get_compute_mode() { return GSH::instance().get_compute_mode(); }
inline void set_compute_mode(Computation mode) { GSH::instance().set_compute_mode(mode); }

inline SpaceTransformation get_space_transformation() { return GSH::instance().get_space_transformation(); }

inline TimeTransformation get_time_transformation() { return GSH::instance().get_time_transformation(); }

inline ImgType get_img_type()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::ImageType>().value;
}
inline void set_img_type(ImgType type)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::ImageType{type});
    return GSH::instance().set_img_type(type);
}

inline WindowKind get_current_window_type() { return GSH::instance().get_current_window_type(); }

inline uint get_batch_size() { return GSH::instance().get_batch_size(); }
inline void set_batch_size(uint value) { GSH::instance().set_batch_size(value); }

inline uint get_time_stride() { return GSH::instance().get_time_stride(); }
inline void set_time_stride(uint value) { GSH::instance().set_time_stride(value); }

inline uint get_time_transformation_size() { return GSH::instance().get_time_transformation_size(); }
inline void set_time_transformation_size(uint value) { GSH::instance().set_time_transformation_size(value); }

inline float get_lambda() { return GSH::instance().get_lambda(); }
inline void set_lambda(float value) { GSH::instance().set_lambda(value); }

inline float get_z_distance() { return GSH::instance().get_z_distance(); }

inline float get_contrast_lower_threshold() { return GSH::instance().get_contrast_lower_threshold(); }
inline void set_contrast_lower_threshold(float value) { GSH::instance().set_contrast_lower_threshold(value); }

inline float get_contrast_upper_threshold() { return GSH::instance().get_contrast_upper_threshold(); }
inline void set_contrast_upper_threshold(float value) { GSH::instance().set_contrast_upper_threshold(value); }

inline uint get_cuts_contrast_p_offset() { return GSH::instance().get_cuts_contrast_p_offset(); }
inline void set_cuts_contrast_p_offset(uint value) { GSH::instance().set_cuts_contrast_p_offset(value); }

inline float get_pixel_size() { return GSH::instance().get_pixel_size(); }

inline unsigned get_renorm_constant() { return GSH::instance().get_renorm_constant(); }
inline void set_renorm_constant(unsigned int value) { GSH::instance().set_renorm_constant(value); }

inline int get_filter2d_n1() { return GSH::instance().get_filter2d_n1(); }
inline void set_filter2d_n1(int value)
{
    GSH::instance().set_filter2d_n1(value);
    set_auto_contrast_all();
}

inline int get_filter2d_n2() { return GSH::instance().get_filter2d_n2(); }
inline void set_filter2d_n2(int value)
{
    GSH::instance().set_filter2d_n2(value);
    set_auto_contrast_all();
}

inline int get_filter2d_smooth_low() { return GSH::instance().get_filter2d_smooth_low(); }
inline void set_filter2d_smooth_low(int value) { GSH::instance().set_filter2d_smooth_low(value); }

inline int get_filter2d_smooth_high() { return GSH::instance().get_filter2d_smooth_high(); }
inline void set_filter2d_smooth_high(int value) { GSH::instance().set_filter2d_smooth_high(value); }

inline float get_display_rate() { return GSH::instance().get_display_rate(); }
inline void set_display_rate(float value) { GSH::instance().set_display_rate(value); }

inline ViewXY get_x(void) { return holovibes::Holovibes::instance().get_setting<settings::X>().value; }

inline uint get_x_cuts() { return holovibes::Holovibes::instance().get_setting<settings::X>().value.start; }

inline int get_x_accu_level() { return holovibes::Holovibes::instance().get_setting<holovibes::settings::X>().value.width; }

inline ViewXY get_y(void) { return holovibes::Holovibes::instance().get_setting<settings::Y>().value; }

inline uint get_y_cuts() { return holovibes::Holovibes::instance().get_setting<settings::Y>().value.start; }

inline int get_y_accu_level() { return holovibes::Holovibes::instance().get_setting<settings::Y>().value.width; }

//XY
inline ViewXYZ get_xy() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value; }
inline bool get_xy_flip_enabled() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.horizontal_flip; }
inline float get_xy_rot() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.rotation; }
inline uint get_xy_img_accu_level() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.output_image_accumulation; }
inline bool get_xy_log_scale_slice_enabled() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.log_enabled; }
inline bool get_xy_contrast_enabled() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.enabled; }
inline bool get_xy_contrast_auto_refresh() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.auto_refresh; }
inline bool get_xy_contrast_invert() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.invert; }
inline float get_xy_contrast_min() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.min; }
inline float get_xy_contrast_max() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.contrast.max; }
inline bool get_xy_img_accu_enabled() { return holovibes::Holovibes::instance().get_setting<settings::XY>().value.output_image_accumulation > 1; }


inline void set_xy(ViewXYZ value) noexcept { holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{value}); }

inline void set_xy_flip_enabled(bool value) noexcept {
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.horizontal_flip = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_rot(float value) noexcept {
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.rotation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_img_accu_level(uint value) {
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.output_image_accumulation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_log_scale_slice_enabled(bool value) noexcept {
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.log_enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast_enabled(bool value) noexcept {
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast_auto_refresh(bool value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.auto_refresh = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast_invert(bool value) noexcept {
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.invert = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast_min(float value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.min = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

inline void set_xy_contrast_max(float value) noexcept
{
    auto xy = Holovibes::instance().get_setting<settings::XY>().value;
    xy.contrast.max = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XY{xy});
}

//XZ
inline ViewXYZ get_xz() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value; }
inline bool get_xz_flip_enabled() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.horizontal_flip; }
inline float get_xz_rot() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.rotation; }
inline uint get_xz_img_accu_level() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.output_image_accumulation; }
inline bool get_xz_log_scale_slice_enabled() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.log_enabled; }
inline bool get_xz_contrast_enabled() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.enabled; }
inline bool get_xz_contrast_auto_refresh() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.auto_refresh; }
inline bool get_xz_contrast_invert() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.invert; }
inline float get_xz_contrast_min() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.min; }
inline float get_xz_contrast_max() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.contrast.max; }
inline bool get_xz_img_accu_enabled() { return holovibes::Holovibes::instance().get_setting<settings::XZ>().value.output_image_accumulation > 1; }
inline void set_xz(ViewXYZ value) noexcept { holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{value}); }

inline void set_xz_flip_enabled(bool value) noexcept {
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.horizontal_flip = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_rot(float value) noexcept {
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.rotation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_img_accu_level(uint value) {
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.output_image_accumulation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_log_scale_slice_enabled(bool value) noexcept {
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.log_enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_enabled(bool value) noexcept {
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_auto_refresh(bool value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.auto_refresh = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_invert(bool value) noexcept {
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.invert = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_min(float value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.min = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline void set_xz_contrast_max(float value) noexcept
{
    auto xz = Holovibes::instance().get_setting<settings::XZ>().value;
    xz.contrast.max = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::XZ{xz});
}

inline ViewXYZ get_yz() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value; }
inline bool get_yz_flip_enabled() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.horizontal_flip; }
inline float get_yz_rot() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.rotation; }
inline uint get_yz_img_accu_level() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.output_image_accumulation; }
inline bool get_yz_log_scale_slice_enabled() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.log_enabled; }
inline bool get_yz_contrast_enabled() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.enabled; }
inline bool get_yz_contrast_auto_refresh() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.auto_refresh; }
inline bool get_yz_contrast_invert() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.invert; }
inline float get_yz_contrast_min() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.min; }
inline float get_yz_contrast_max() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.contrast.max; }
inline bool get_yz_img_accu_enabled() { return holovibes::Holovibes::instance().get_setting<settings::YZ>().value.output_image_accumulation > 1; }
inline void set_yz(ViewXYZ value) noexcept { holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{value}); }

inline void set_yz_flip_enabled(bool value) noexcept {
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.horizontal_flip = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_rot(float value) noexcept {
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.rotation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_img_accu_level(uint value) {
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.output_image_accumulation = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_log_scale_slice_enabled(bool value) noexcept {
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.log_enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_enabled(bool value) noexcept {
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.enabled = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_auto_refresh(bool value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.auto_refresh = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_invert(bool value) noexcept {
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.invert = value;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_min(float value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.min = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}

inline void set_yz_contrast_max(float value) noexcept
{
    auto yz = Holovibes::instance().get_setting<settings::YZ>().value;
    yz.contrast.max = value > 1.0f ? value : 1.0f;
    holovibes::Holovibes::instance().update_setting(holovibes::settings::YZ{yz});
}



inline float get_reticle_scale() { return GSH::instance().get_reticle_scale(); }
inline void set_reticle_scale(float value) { GSH::instance().set_reticle_scale(value); }

inline CompositeKind get_composite_kind() { return GSH::instance().get_composite_kind(); }
inline void set_composite_kind(CompositeKind value) { GSH::instance().set_composite_kind(value); }

inline bool get_flip_enabled() { return GSH::instance().get_flip_enabled(); }

inline double get_rotation() { return GSH::instance().get_rotation(); }

// RGB
inline uint get_composite_p_red() { return GSH::instance().get_rgb_p_min(); }
inline uint get_composite_p_blue() { return GSH::instance().get_rgb_p_max(); }
inline void set_rgb_p(int min, int max) { GSH::instance().set_rgb_p({min, max}, true); }

inline float get_weight_r() { return GSH::instance().get_weight_r(); }

inline float get_weight_g() { return GSH::instance().get_weight_g(); }

inline float get_weight_b() { return GSH::instance().get_weight_b(); }

// HSV
inline uint get_composite_p_min_h() { return GSH::instance().get_composite_p_min_h(); }
inline uint get_composite_p_max_h() { return GSH::instance().get_composite_p_max_h(); }
inline void set_composite_p_h(unsigned int min, unsigned int max)
{
    GSH::instance().set_composite_p_h({min, max}, true);
}

inline float get_slider_h_threshold_min() { return GSH::instance().get_slider_h_threshold_min(); }
inline void set_slider_h_threshold_min(float value) { GSH::instance().set_slider_h_threshold_min(value); }

inline float get_slider_h_threshold_max() { return GSH::instance().get_slider_h_threshold_max(); }
inline void set_slider_h_threshold_max(float value) { GSH::instance().set_slider_h_threshold_max(value); }

inline float get_composite_low_h_threshold() { return GSH::instance().get_composite_low_h_threshold(); }

inline float get_composite_high_h_threshold() { return GSH::instance().get_composite_high_h_threshold(); }

inline uint get_h_blur_kernel_size() { return GSH::instance().get_h_blur_kernel_size(); }

inline uint get_composite_p_min_s() { return GSH::instance().get_composite_p_min_s(); }
inline uint get_composite_p_max_s() { return GSH::instance().get_composite_p_max_s(); }

inline float get_slider_s_threshold_min() { return GSH::instance().get_slider_s_threshold_min(); }
inline void set_slider_s_threshold_min(float value) { GSH::instance().set_slider_s_threshold_min(value); }

inline float get_slider_s_threshold_max() { return GSH::instance().get_slider_s_threshold_max(); }
inline void set_slider_s_threshold_max(float value) { GSH::instance().set_slider_s_threshold_max(value); }

inline float get_composite_low_s_threshold() { return GSH::instance().get_composite_low_s_threshold(); }

inline float get_composite_high_s_threshold() { return GSH::instance().get_composite_high_s_threshold(); }

inline uint get_composite_p_min_v() { return GSH::instance().get_composite_p_min_v(); }

inline uint get_composite_p_max_v() { return GSH::instance().get_composite_p_max_v(); }

inline float get_slider_v_threshold_min() { return GSH::instance().get_slider_v_threshold_min(); }
inline void set_slider_v_threshold_min(float value) { GSH::instance().set_slider_v_threshold_min(value); }

inline float get_slider_v_threshold_max() { return GSH::instance().get_slider_v_threshold_max(); }
inline void set_slider_v_threshold_max(float value) { GSH::instance().set_slider_v_threshold_max(value); }

inline float get_composite_low_v_threshold() { return GSH::instance().get_composite_low_v_threshold(); }

inline float get_composite_high_v_threshold() { return GSH::instance().get_composite_high_v_threshold(); }

inline int get_unwrap_history_size() { return GSH::instance().get_unwrap_history_size(); }

inline bool get_is_computation_stopped() { return GSH::instance().get_is_computation_stopped(); }
inline void set_is_computation_stopped(bool value) { GSH::instance().set_is_computation_stopped(value); }

inline bool get_convolution_enabled() { return GSH::instance().get_convolution_enabled(); }

inline bool get_divide_convolution_enabled() { return GSH::instance().get_divide_convolution_enabled(); }
inline void set_divide_convolution_enabled(bool value) { return GSH::instance().set_divide_convolution_enabled(value); }

inline bool get_renorm_enabled() { return GSH::instance().get_renorm_enabled(); }
inline void set_renorm_enabled(bool value) { GSH::instance().set_renorm_enabled(value); }

inline bool get_fft_shift_enabled() { return GSH::instance().get_fft_shift_enabled(); }
inline void set_fft_shift_enabled(bool value) { GSH::instance().set_fft_shift_enabled(value); }

inline bool get_log_scale_slice_xy_enabled() { return GSH::instance().get_xy_log_scale_slice_enabled(); }

inline bool get_log_scale_slice_xz_enabled() { return GSH::instance().get_xz_log_scale_slice_enabled(); }

inline bool get_log_scale_slice_yz_enabled() { return GSH::instance().get_yz_log_scale_slice_enabled(); }

inline bool get_contrast_enabled() { return GSH::instance().get_contrast_enabled(); }

inline bool get_contrast_auto_refresh() { return GSH::instance().get_contrast_auto_refresh(); }

inline bool get_contrast_invert() { return GSH::instance().get_contrast_invert(); }

inline bool get_filter2d_enabled() { return GSH::instance().get_filter2d_enabled(); }
inline void set_filter2d_enabled(bool value) { return GSH::instance().set_filter2d_enabled(value); }

inline bool get_filter2d_view_enabled() { return GSH::instance().get_filter2d_view_enabled(); }

inline bool get_cuts_view_enabled() { return GSH::instance().get_cuts_view_enabled(); }
inline void set_cuts_view_enabled(bool value) { GSH::instance().set_cuts_view_enabled(value); }

inline bool get_lens_view_enabled() { return GSH::instance().get_lens_view_enabled(); }
inline void set_lens_view_enabled(bool value) { GSH::instance().set_lens_view_enabled(value); }

inline bool get_chart_display_enabled() { return GSH::instance().get_chart_display_enabled(); }
inline bool get_chart_record_enabled() { return GSH::instance().get_chart_record_enabled(); }

inline bool get_raw_view_enabled() { return GSH::instance().get_raw_view_enabled(); }

inline bool get_reticle_display_enabled() { return GSH::instance().get_reticle_display_enabled(); }
inline void set_reticle_display_enabled(bool value) { GSH::instance().set_reticle_display_enabled(value); }

inline bool get_h_blur_activated() { return GSH::instance().get_h_blur_activated(); }

inline bool get_composite_p_activated_s() { return GSH::instance().get_composite_p_activated_s(); }

inline bool get_composite_p_activated_v() { return GSH::instance().get_composite_p_activated_v(); }

inline bool get_composite_auto_weights() { return GSH::instance().get_composite_auto_weights(); }
inline void set_composite_auto_weights(bool value) { GSH::instance().set_composite_auto_weights(value); }

inline uint get_file_buffer_size() { return GSH::instance().get_file_buffer_size(); }
inline void set_file_buffer_size(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::FileBufferSize{value});
    GSH::instance().set_file_buffer_size(value);
}

inline uint get_input_buffer_size() { return GSH::instance().get_input_buffer_size(); }
inline void set_input_buffer_size(uint value) { GSH::instance().set_input_buffer_size(value); }

// inline uint get_output_buffer_size() { return GSH::instance().get_output_buffer_size(); }
// inline void set_output_buffer_size(uint value) { GSH::instance().set_output_buffer_size(value); }

inline uint get_record_buffer_size() { return GSH::instance().get_record_buffer_size(); }
inline void set_record_buffer_size(uint value) { GSH::instance().set_record_buffer_size(value); }

inline uint get_time_transformation_cuts_output_buffer_size()
{
    return GSH::instance().get_time_transformation_cuts_output_buffer_size();
}
inline void set_time_transformation_cuts_output_buffer_size(uint value)
{
    GSH::instance().set_time_transformation_cuts_output_buffer_size(value);
}

inline bool get_batch_enabled()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::BatchEnabled>().value;
}

inline void set_batch_enabled(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::BatchEnabled{value});
}

inline std::optional<std::string> get_batch_file_path()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::BatchFilePath>().value;
}

inline void set_batch_file_path(std::string value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::BatchFilePath{value});
}

inline std::string get_record_file_path()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::RecordFilePath>().value;
}

inline void set_record_file_path(std::string value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RecordFilePath{value});
}

inline std::optional<size_t> get_record_frame_count()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::RecordFrameCount>().value;
}

inline void set_record_frame_count(std::optional<size_t> value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RecordFrameCount{value});
}

inline RecordMode get_record_mode()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::RecordMode>().value;
}

inline void set_record_mode(RecordMode value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RecordMode{value});
}

inline size_t get_record_frame_skip()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::RecordFrameSkip>().value;
}

inline void set_record_frame_skip(size_t value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::RecordFrameSkip{value});
}

inline size_t get_output_buffer_size()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::OutputBufferSize>().value;
}

inline void set_output_buffer_size(size_t value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::OutputBufferSize{value});
}

inline uint get_input_fps()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::InputFPS>().value;
}

inline void set_input_fps(uint value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::InputFPS{value});
}

inline std::string get_input_file_path()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::InputFilePath>().value;
}

inline void set_input_file_path(std::string value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::InputFilePath{value});
}

inline bool get_loop_on_input_file()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::LoopOnInputFile>().value;
}

inline void set_loop_on_input_file(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::LoopOnInputFile{value});
}

inline bool get_load_file_in_gpu()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::LoadFileInGPU>().value;
}

inline void set_load_file_in_gpu(bool value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::LoadFileInGPU{value});
}

inline size_t get_input_file_start_index()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::InputFileStartIndex>().value;
}

inline void set_input_file_start_index(size_t value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::InputFileStartIndex{value});
}

inline size_t get_input_file_end_index()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::InputFileEndIndex>().value;
}

inline void set_input_file_end_index(size_t value)
{
    holovibes::Holovibes::instance().update_setting(holovibes::settings::InputFileEndIndex{value});
}

inline ViewPQ get_p()
{ 
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::P>().value; 
}

inline int get_p_accu_level() 
{ 
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::P>().value.width; 
}

inline uint get_p_index()
{ 
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::P>().value.start; 
}

inline ViewPQ get_q(void) 
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::Q>().value;
}

inline uint get_q_index()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::Q>().value.start;
}

inline uint get_q_accu_level()
{
    return holovibes::Holovibes::instance().get_setting<holovibes::settings::Q>().value.width;
}

inline const camera::FrameDescriptor& get_fd() { return Holovibes::instance().get_gpu_input_queue()->get_fd(); };

inline std::shared_ptr<Pipe> get_compute_pipe() { return Holovibes::instance().get_compute_pipe(); };

inline std::shared_ptr<Queue> get_gpu_output_queue() { return Holovibes::instance().get_gpu_output_queue(); };

inline std::shared_ptr<BatchInputQueue> get_gpu_input_queue() { return Holovibes::instance().get_gpu_input_queue(); };

inline units::RectFd get_signal_zone() { return GSH::instance().get_signal_zone(); };
inline units::RectFd get_noise_zone() { return GSH::instance().get_noise_zone(); };
inline units::RectFd get_composite_zone() { return GSH::instance().get_composite_zone(); };
inline units::RectFd get_zoomed_zone() { return GSH::instance().get_zoomed_zone(); };
inline units::RectFd get_reticle_zone() { return GSH::instance().get_reticle_zone(); };

inline void set_signal_zone(const units::RectFd& rect) { GSH::instance().set_signal_zone(rect); };
inline void set_noise_zone(const units::RectFd& rect) { GSH::instance().set_noise_zone(rect); };
inline void set_composite_zone(const units::RectFd& rect) { GSH::instance().set_composite_zone(rect); };
inline void set_zoomed_zone(const units::RectFd& rect) { GSH::instance().set_zoomed_zone(rect); };
inline void set_reticle_zone(const units::RectFd& rect) { GSH::instance().set_reticle_zone(rect); };

#pragma endregion

} // namespace holovibes::api
