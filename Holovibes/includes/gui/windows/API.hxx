#pragma once

#include "API.hh"

namespace holovibes::api
{

inline Computation get_compute_mode() { return Holovibes::instance().get_cd().compute_mode; }
inline void set_compute_mode(Computation compute_mode) { Holovibes::instance().get_cd().compute_mode = compute_mode; }

inline SpaceTransformation get_space_transformation() { return Holovibes::instance().get_cd().space_transformation; }
inline void set_space_transformation(SpaceTransformation space_transformation)
{
    Holovibes::instance().get_cd().space_transformation = space_transformation;
}

inline TimeTransformation get_time_transformation() { return Holovibes::instance().get_cd().time_transformation; }
inline void set_time_transformation(TimeTransformation time_transformation)
{
    Holovibes::instance().get_cd().time_transformation = time_transformation;
}

inline ImgType get_img_type() { return Holovibes::instance().get_cd().img_type; }
inline void set_img_type(ImgType img_type) { Holovibes::instance().get_cd().img_type = img_type; }

inline WindowKind get_current_window() { return Holovibes::instance().get_cd().current_window; }
inline void set_current_window(WindowKind current_window)
{
    Holovibes::instance().get_cd().current_window = current_window;
}

inline uint get_batch_size() { return Holovibes::instance().get_cd().batch_size; }
inline void set_batch_size(uint batch_size) { Holovibes::instance().get_cd().batch_size = batch_size; }

inline uint get_time_transformation_stride() { return Holovibes::instance().get_cd().time_transformation_stride; }
inline void set_time_transformation_stride(uint time_transformation_stride)
{
    Holovibes::instance().get_cd().time_transformation_stride = time_transformation_stride;
}

inline uint get_time_transformation_size() { return Holovibes::instance().get_cd().time_transformation_size; }
inline void set_time_transformation_size(uint time_transformation_size)
{
    Holovibes::instance().get_cd().time_transformation_size = time_transformation_size;
}

inline float get_lambda() { return Holovibes::instance().get_cd().lambda; }
inline void set_lambda(float lambda) { Holovibes::instance().get_cd().lambda = lambda; }

inline float get_zdistance() { return Holovibes::instance().get_cd().zdistance; }
inline void set_zdistance(float zdistance) { Holovibes::instance().get_cd().zdistance = zdistance; }

inline float get_contrast_min_slice_xy() { return Holovibes::instance().get_cd().contrast_min_slice_xy; }
inline void set_contrast_min_slice_xy(float contrast_min_slice_xy)
{
    Holovibes::instance().get_cd().contrast_min_slice_xy = contrast_min_slice_xy;
}

inline float get_contrast_max_slice_xy() { return Holovibes::instance().get_cd().contrast_max_slice_xy; }
inline void set_contrast_max_slice_xy(float contrast_max_slice_xy)
{
    Holovibes::instance().get_cd().contrast_max_slice_xy = contrast_max_slice_xy;
}

inline float get_contrast_min_slice_xz() { return Holovibes::instance().get_cd().contrast_min_slice_xz; }
inline void set_contrast_min_slice_xz(float contrast_min_slice_xz)
{
    Holovibes::instance().get_cd().contrast_min_slice_xz = contrast_min_slice_xz;
}

inline float get_contrast_max_slice_xz() { return Holovibes::instance().get_cd().contrast_max_slice_xz; }
inline void set_contrast_max_slice_xz(float contrast_max_slice_xz)
{
    Holovibes::instance().get_cd().contrast_max_slice_xz = contrast_max_slice_xz;
}

inline float get_contrast_min_slice_yz() { return Holovibes::instance().get_cd().contrast_min_slice_yz; }
inline void set_contrast_min_slice_yz(float contrast_min_slice_yz)
{
    Holovibes::instance().get_cd().contrast_min_slice_yz = contrast_min_slice_yz;
}

inline float get_contrast_max_slice_yz() { return Holovibes::instance().get_cd().contrast_max_slice_yz; }
inline void set_contrast_max_slice_yz(float contrast_max_slice_yz)
{
    Holovibes::instance().get_cd().contrast_max_slice_yz = contrast_max_slice_yz;
}

inline float get_contrast_min_filter2d() { return Holovibes::instance().get_cd().contrast_min_filter2d; }
inline void set_contrast_min_filter2d(float contrast_min_filter2d)
{
    Holovibes::instance().get_cd().contrast_min_filter2d = contrast_min_filter2d;
}

inline float get_contrast_max_filter2d() { return Holovibes::instance().get_cd().contrast_max_filter2d; }
inline void set_contrast_max_filter2d(float contrast_max_filter2d)
{
    Holovibes::instance().get_cd().contrast_max_filter2d = contrast_max_filter2d;
}

inline float get_contrast_lower_threshold() { return Holovibes::instance().get_cd().contrast_lower_threshold; }
inline void set_contrast_lower_threshold(float contrast_lower_threshold)
{
    Holovibes::instance().get_cd().contrast_lower_threshold = contrast_lower_threshold;
}
inline float get_contrast_upper_threshold() { return Holovibes::instance().get_cd().contrast_upper_threshold; }
inline void set_contrast_upper_threshold(float contrast_upper_threshold)
{
    Holovibes::instance().get_cd().contrast_upper_threshold = contrast_upper_threshold;
}
inline uint get_cuts_contrast_p_offset() { return Holovibes::instance().get_cd().cuts_contrast_p_offset; }
inline void set_cuts_contrast_p_offset(uint cuts_contrast_p_offset)
{
    Holovibes::instance().get_cd().cuts_contrast_p_offset = cuts_contrast_p_offset;
}

inline float get_pixel_size() { return Holovibes::instance().get_cd().pixel_size; }
inline void set_pixel_size(float pixel_size) { Holovibes::instance().get_cd().pixel_size = pixel_size; }

inline unsigned get_renorm_constant() { return Holovibes::instance().get_cd().renorm_constant; }
inline void set_renorm_constant(unsigned renorm_constant)
{
    Holovibes::instance().get_cd().renorm_constant = renorm_constant;
}

inline int get_filter2d_n1() { return Holovibes::instance().get_cd().filter2d_n1; }
inline void set_filter2d_n1(int filter2d_n1) { Holovibes::instance().get_cd().filter2d_n1 = filter2d_n1; }

inline int get_filter2d_n2() { return Holovibes::instance().get_cd().filter2d_n2; }
inline void set_filter2d_n2(int filter2d_n2) { Holovibes::instance().get_cd().filter2d_n2 = filter2d_n2; }

inline int get_filter2d_smooth_low() { return Holovibes::instance().get_cd().filter2d_smooth_low; }
inline void set_filter2d_smooth_low(int filter2d_smooth_low)
{
    Holovibes::instance().get_cd().filter2d_smooth_low = filter2d_smooth_low;
}

inline int get_filter2d_smooth_high() { return Holovibes::instance().get_cd().filter2d_smooth_high; }
inline void set_filter2d_smooth_high(int filter2d_smooth_high)
{
    Holovibes::instance().get_cd().filter2d_smooth_high = filter2d_smooth_high;
}

inline float get_display_rate() { return Holovibes::instance().get_cd().display_rate; }
inline void set_display_rate(float display_rate) { Holovibes::instance().get_cd().display_rate = display_rate; }

inline uint get_img_acc_slice_xy_level() { return Holovibes::instance().get_cd().img_acc_slice_xy_level; }
inline void set_img_acc_slice_xy_level(uint img_acc_slice_xy_level)
{
    Holovibes::instance().get_cd().img_acc_slice_xy_level = img_acc_slice_xy_level;
}

inline uint get_img_acc_slice_xz_level() { return Holovibes::instance().get_cd().img_acc_slice_xz_level; }
inline void set_img_acc_slice_xz_level(uint img_acc_slice_xz_level)
{
    Holovibes::instance().get_cd().img_acc_slice_xz_level = img_acc_slice_xz_level;
}

inline uint get_img_acc_slice_yz_level() { return Holovibes::instance().get_cd().img_acc_slice_yz_level; }
inline void set_img_acc_slice_yz_level(uint img_acc_slice_yz_level)
{
    Holovibes::instance().get_cd().img_acc_slice_yz_level = img_acc_slice_yz_level;
}

inline uint get_pindex() { return Holovibes::instance().get_cd().pindex; }
inline void set_pindex(uint pindex) { Holovibes::instance().get_cd().pindex = pindex; }

inline int get_p_acc_level() { return Holovibes::instance().get_cd().p_acc_level; }
inline void set_p_acc_level(int p_acc_level) { Holovibes::instance().get_cd().p_acc_level = p_acc_level; }

inline uint get_x_cuts() { return Holovibes::instance().get_cd().x_cuts; }
inline void set_x_cuts(uint x_cuts) { Holovibes::instance().get_cd().x_cuts = x_cuts; }

inline int get_x_acc_level() { return Holovibes::instance().get_cd().x_acc_level; }
inline void set_x_acc_level(int x_acc_level) { Holovibes::instance().get_cd().x_acc_level = x_acc_level; }

inline uint get_y_cuts() { return Holovibes::instance().get_cd().y_cuts; }
inline void set_y_cuts(uint y_cuts) { Holovibes::instance().get_cd().y_cuts = y_cuts; }

inline int get_y_acc_level() { return Holovibes::instance().get_cd().y_acc_level; }
inline void set_y_acc_level(int y_acc_level) { Holovibes::instance().get_cd().y_acc_level = y_acc_level; }

inline uint get_q_index() { return Holovibes::instance().get_cd().q_index; }
inline void set_q_index(uint q_index) { Holovibes::instance().get_cd().q_index = q_index; }

inline uint get_q_acc_level() { return Holovibes::instance().get_cd().q_acc_level; }
inline void set_q_acc_level(uint q_acc_level) { Holovibes::instance().get_cd().q_acc_level = q_acc_level; }

inline float get_reticle_scale() { return Holovibes::instance().get_cd().reticle_scale; }
inline void set_reticle_scale(float reticle_scale) { Holovibes::instance().get_cd().reticle_scale = reticle_scale; }

inline uint get_raw_bitshift() { return Holovibes::instance().get_cd().raw_bitshift; }
inline void set_raw_bitshift(uint raw_bitshift) { Holovibes::instance().get_cd().raw_bitshift = raw_bitshift; }

inline CompositeKind get_composite_kind() { return Holovibes::instance().get_cd().composite_kind; }
inline void set_composite_kind(CompositeKind composite_kind)
{
    Holovibes::instance().get_cd().composite_kind = composite_kind;
}

// RGB
inline uint get_composite_p_red() { return Holovibes::instance().get_cd().composite_p_red; }
inline void set_composite_p_red(uint composite_p_red)
{
    Holovibes::instance().get_cd().composite_p_red = composite_p_red;
}
inline uint get_composite_p_blue() { return Holovibes::instance().get_cd().composite_p_blue; }
inline void set_composite_p_blue(uint composite_p_blue)
{
    Holovibes::instance().get_cd().composite_p_blue = composite_p_blue;
}
inline float get_weight_r() { return Holovibes::instance().get_cd().weight_r; }
inline void set_weight_r(float weight_r) { Holovibes::instance().get_cd().weight_r = weight_r; }
inline float get_weight_g() { return Holovibes::instance().get_cd().weight_g; }
inline void set_weight_g(float weight_g) { Holovibes::instance().get_cd().weight_g = weight_g; }
inline float get_weight_b() { return Holovibes::instance().get_cd().weight_b; }
inline void set_weight_b(float weight_b) { Holovibes::instance().get_cd().weight_b = weight_b; }

// HSV
inline uint get_composite_p_min_h() { return Holovibes::instance().get_cd().composite_p_min_h; }
inline void set_composite_p_min_h(uint composite_p_min_h)
{
    Holovibes::instance().get_cd().composite_p_min_h = composite_p_min_h;
}
inline uint get_composite_p_max_h() { return Holovibes::instance().get_cd().composite_p_max_h; }
inline void set_composite_p_max_h(uint composite_p_max_h)
{
    Holovibes::instance().get_cd().composite_p_max_h = composite_p_max_h;
}
inline float get_slider_h_threshold_min() { return Holovibes::instance().get_cd().slider_h_threshold_min; }
inline void set_slider_h_threshold_min(float slider_h_threshold_min)
{
    Holovibes::instance().get_cd().slider_h_threshold_min = slider_h_threshold_min;
}
inline float get_slider_h_threshold_max() { return Holovibes::instance().get_cd().slider_h_threshold_max; }
inline void set_slider_h_threshold_max(float slider_h_threshold_max)
{
    Holovibes::instance().get_cd().slider_h_threshold_max = slider_h_threshold_max;
}
inline float get_composite_low_h_threshold() { return Holovibes::instance().get_cd().composite_low_h_threshold; }
inline void set_composite_low_h_threshold(float composite_low_h_threshold)
{
    Holovibes::instance().get_cd().composite_low_h_threshold = composite_low_h_threshold;
}
inline float get_composite_high_h_threshold() { return Holovibes::instance().get_cd().composite_high_h_threshold; }
inline void set_composite_high_h_threshold(float composite_high_h_threshold)
{
    Holovibes::instance().get_cd().composite_high_h_threshold = composite_high_h_threshold;
}
inline uint get_h_blur_kernel_size() { return Holovibes::instance().get_cd().h_blur_kernel_size; }
inline void set_h_blur_kernel_size(uint h_blur_kernel_size)
{
    Holovibes::instance().get_cd().h_blur_kernel_size = h_blur_kernel_size;
}

inline uint get_composite_p_min_s() { return Holovibes::instance().get_cd().composite_p_min_s; }
inline void set_composite_p_min_s(uint composite_p_min_s)
{
    Holovibes::instance().get_cd().composite_p_min_s = composite_p_min_s;
}
inline uint get_composite_p_max_s() { return Holovibes::instance().get_cd().composite_p_max_s; }
inline void set_composite_p_max_s(uint composite_p_max_s)
{
    Holovibes::instance().get_cd().composite_p_max_s = composite_p_max_s;
}
inline float get_slider_s_threshold_min() { return Holovibes::instance().get_cd().slider_s_threshold_min; }
inline void set_slider_s_threshold_min(float slider_s_threshold_min)
{
    Holovibes::instance().get_cd().slider_s_threshold_min = slider_s_threshold_min;
}
inline float get_slider_s_threshold_max() { return Holovibes::instance().get_cd().slider_s_threshold_max; }
inline void set_slider_s_threshold_max(float slider_s_threshold_max)
{
    Holovibes::instance().get_cd().slider_s_threshold_max = slider_s_threshold_max;
}
inline float get_composite_low_s_threshold() { return Holovibes::instance().get_cd().composite_low_s_threshold; }
inline void set_composite_low_s_threshold(float composite_low_s_threshold)
{
    Holovibes::instance().get_cd().composite_low_s_threshold = composite_low_s_threshold;
}
inline float get_composite_high_s_threshold() { return Holovibes::instance().get_cd().composite_high_s_threshold; }
inline void set_composite_high_s_threshold(float composite_high_s_threshold)
{
    Holovibes::instance().get_cd().composite_high_s_threshold = composite_high_s_threshold;
}

inline uint get_composite_p_min_v() { return Holovibes::instance().get_cd().composite_p_min_v; }
inline void set_composite_p_min_v(uint composite_p_min_v)
{
    Holovibes::instance().get_cd().composite_p_min_v = composite_p_min_v;
}
inline uint get_composite_p_max_v() { return Holovibes::instance().get_cd().composite_p_max_v; }
inline void set_composite_p_max_v(uint composite_p_max_v)
{
    Holovibes::instance().get_cd().composite_p_max_v = composite_p_max_v;
}
inline float get_slider_v_threshold_min() { return Holovibes::instance().get_cd().slider_v_threshold_min; }
inline void set_slider_v_threshold_min(float slider_v_threshold_min)
{
    Holovibes::instance().get_cd().slider_v_threshold_min = slider_v_threshold_min;
}
inline float get_slider_v_threshold_max() { return Holovibes::instance().get_cd().slider_v_threshold_max; }
inline void set_slider_v_threshold_max(float slider_v_threshold_max)
{
    Holovibes::instance().get_cd().slider_v_threshold_max = slider_v_threshold_max;
}
inline float get_composite_low_v_threshold() { return Holovibes::instance().get_cd().composite_low_v_threshold; }
inline void set_composite_low_v_threshold(float composite_low_v_threshold)
{
    Holovibes::instance().get_cd().composite_low_v_threshold = composite_low_v_threshold;
}
inline float get_composite_high_v_threshold() { return Holovibes::instance().get_cd().composite_high_v_threshold; }
inline void set_composite_high_v_threshold(float composite_high_v_threshold)
{
    Holovibes::instance().get_cd().composite_high_v_threshold = composite_high_v_threshold;
}

inline int get_unwrap_history_size() { return Holovibes::instance().get_cd().unwrap_history_size; }
inline void set_unwrap_history_size(int unwrap_history_size)
{
    Holovibes::instance().get_cd().unwrap_history_size = unwrap_history_size;
}

inline bool get_is_computation_stopped() { return Holovibes::instance().get_cd().is_computation_stopped; }
inline void set_is_computation_stopped(bool is_computation_stopped)
{
    Holovibes::instance().get_cd().is_computation_stopped = is_computation_stopped;
}

inline bool get_convolution_enabled() { return Holovibes::instance().get_cd().convolution_enabled; }
inline void set_convolution_enabled(bool convolution_enabled)
{
    Holovibes::instance().get_cd().convolution_enabled = convolution_enabled;
}

inline bool get_divide_convolution_enabled() { return Holovibes::instance().get_cd().divide_convolution_enabled; }
inline void set_divide_convolution_enabled(bool divide_convolution_enabled)
{
    Holovibes::instance().get_cd().divide_convolution_enabled = divide_convolution_enabled;
}

inline bool get_renorm_enabled() { return Holovibes::instance().get_cd().renorm_enabled; }
inline void set_renorm_enabled(bool renorm_enabled) { Holovibes::instance().get_cd().renorm_enabled = renorm_enabled; }

inline bool get_fft_shift_enabled() { return Holovibes::instance().get_cd().fft_shift_enabled; }
inline void set_fft_shift_enabled(bool fft_shift_enabled)
{
    Holovibes::instance().get_cd().fft_shift_enabled = fft_shift_enabled;
}

inline bool get_frame_record_enabled() { return Holovibes::instance().get_cd().frame_record_enabled; }
inline void set_frame_record_enabled(bool frame_record_enabled)
{
    Holovibes::instance().get_cd().frame_record_enabled = frame_record_enabled;
}

inline bool get_log_scale_slice_xy_enabled() { return Holovibes::instance().get_cd().log_scale_slice_xy_enabled; }
inline void set_log_scale_slice_xy_enabled(bool log_scale_slice_xy_enabled)
{
    Holovibes::instance().get_cd().log_scale_slice_xy_enabled = log_scale_slice_xy_enabled;
}

inline bool get_log_scale_slice_xz_enabled() { return Holovibes::instance().get_cd().log_scale_slice_xz_enabled; }
inline void set_log_scale_slice_xz_enabled(bool log_scale_slice_xz_enabled)
{
    Holovibes::instance().get_cd().log_scale_slice_xz_enabled = log_scale_slice_xz_enabled;
}

inline bool get_log_scale_slice_yz_enabled() { return Holovibes::instance().get_cd().log_scale_slice_yz_enabled; }
inline void set_log_scale_slice_yz_enabled(bool log_scale_slice_yz_enabled)
{
    Holovibes::instance().get_cd().log_scale_slice_yz_enabled = log_scale_slice_yz_enabled;
}

inline bool get_log_scale_filter2d_enabled() { return Holovibes::instance().get_cd().log_scale_filter2d_enabled; }
inline void set_log_scale_filter2d_enabled(bool log_scale_filter2d_enabled)
{
    Holovibes::instance().get_cd().log_scale_filter2d_enabled = log_scale_filter2d_enabled;
}

inline bool get_contrast_enabled() { return Holovibes::instance().get_cd().contrast_enabled; }
inline void set_contrast_enabled(bool contrast_enabled)
{
    Holovibes::instance().get_cd().contrast_enabled = contrast_enabled;
}

inline bool get_contrast_auto_refresh() { return Holovibes::instance().get_cd().contrast_auto_refresh; }
inline void set_contrast_auto_refresh(bool contrast_auto_refresh)
{
    Holovibes::instance().get_cd().contrast_auto_refresh = contrast_auto_refresh;
}

inline bool get_contrast_invert() { return Holovibes::instance().get_cd().contrast_invert; }
inline void set_contrast_invert(bool contrast_invert)
{
    Holovibes::instance().get_cd().contrast_invert = contrast_invert;
}

inline bool get_filter2d_enabled() { return Holovibes::instance().get_cd().filter2d_enabled; }
inline void set_filter2d_enabled(bool filter2d_enabled)
{
    Holovibes::instance().get_cd().filter2d_enabled = filter2d_enabled;
}

inline bool get_filter2d_view_enabled() { return Holovibes::instance().get_cd().filter2d_view_enabled; }
inline void set_filter2d_view_enabled(bool filter2d_view_enabled)
{
    Holovibes::instance().get_cd().filter2d_view_enabled = filter2d_view_enabled;
}

inline bool get_time_transformation_cuts_enabled()
{
    return Holovibes::instance().get_cd().time_transformation_cuts_enabled;
}
inline void set_time_transformation_cuts_enabled(bool time_transformation_cuts_enabled)
{
    Holovibes::instance().get_cd().time_transformation_cuts_enabled = time_transformation_cuts_enabled;
}

inline bool get_gpu_lens_display_enabled() { return Holovibes::instance().get_cd().gpu_lens_display_enabled; }
inline void set_gpu_lens_display_enabled(bool gpu_lens_display_enabled)
{
    Holovibes::instance().get_cd().gpu_lens_display_enabled = gpu_lens_display_enabled;
}

inline bool get_chart_display_enabled() { return Holovibes::instance().get_cd().chart_display_enabled; }
inline void set_chart_display_enabled(bool chart_display_enabled)
{
    Holovibes::instance().get_cd().chart_display_enabled = chart_display_enabled;
}

inline bool get_chart_record_enabled() { return Holovibes::instance().get_cd().chart_record_enabled; }
inline void set_chart_record_enabled(bool chart_record_enabled)
{
    Holovibes::instance().get_cd().chart_record_enabled = chart_record_enabled;
}

inline bool get_img_acc_slice_xy_enabled() { return Holovibes::instance().get_cd().img_acc_slice_xy_enabled; }
inline void set_img_acc_slice_xy_enabled(bool img_acc_slice_xy_enabled)
{
    Holovibes::instance().get_cd().img_acc_slice_xy_enabled = img_acc_slice_xy_enabled;
}

inline bool get_img_acc_slice_xz_enabled() { return Holovibes::instance().get_cd().img_acc_slice_xz_enabled; }
inline void set_img_acc_slice_xz_enabled(bool img_acc_slice_xz_enabled)
{
    Holovibes::instance().get_cd().img_acc_slice_xz_enabled = img_acc_slice_xz_enabled;
}

inline bool get_img_acc_slice_yz_enabled() { return Holovibes::instance().get_cd().img_acc_slice_yz_enabled; }
inline void set_img_acc_slice_yz_enabled(bool img_acc_slice_yz_enabled)
{
    Holovibes::instance().get_cd().img_acc_slice_yz_enabled = img_acc_slice_yz_enabled;
}

inline bool get_p_accu_enabled() { return Holovibes::instance().get_cd().p_accu_enabled; }
inline void set_p_accu_enabled(bool p_accu_enabled) { Holovibes::instance().get_cd().p_accu_enabled = p_accu_enabled; }

inline bool get_x_accu_enabled() { return Holovibes::instance().get_cd().x_accu_enabled; }
inline void set_x_accu_enabled(bool x_accu_enabled) { Holovibes::instance().get_cd().x_accu_enabled = x_accu_enabled; }

inline bool get_y_accu_enabled() { return Holovibes::instance().get_cd().y_accu_enabled; }
inline void set_y_accu_enabled(bool y_accu_enabled) { Holovibes::instance().get_cd().y_accu_enabled = y_accu_enabled; }

inline bool get_q_acc_enabled() { return Holovibes::instance().get_cd().q_acc_enabled; }
inline void set_q_acc_enabled(bool q_acc_enabled) { Holovibes::instance().get_cd().q_acc_enabled = q_acc_enabled; }

inline bool get_raw_view_enabled() { return Holovibes::instance().get_cd().raw_view_enabled; }
inline void set_raw_view_enabled(bool raw_view_enabled)
{
    Holovibes::instance().get_cd().raw_view_enabled = raw_view_enabled;
}

inline bool get_synchronized_record() { return Holovibes::instance().get_cd().synchronized_record; }
inline void set_synchronized_record(bool synchronized_record)
{
    Holovibes::instance().get_cd().synchronized_record = synchronized_record;
}

inline bool get_reticle_enabled() { return Holovibes::instance().get_cd().reticle_enabled; }
inline void set_reticle_enabled(bool reticle_enabled)
{
    Holovibes::instance().get_cd().reticle_enabled = reticle_enabled;
}

inline bool get_h_blur_activated() { return Holovibes::instance().get_cd().h_blur_activated; }
inline void set_h_blur_activated(bool h_blur_activated)
{
    Holovibes::instance().get_cd().h_blur_activated = h_blur_activated;
}
inline bool get_composite_p_activated_s() { return Holovibes::instance().get_cd().composite_p_activated_s; }
inline void set_composite_p_activated_s(bool composite_p_activated_s)
{
    Holovibes::instance().get_cd().composite_p_activated_s = composite_p_activated_s;
}
inline bool get_composite_p_activated_v() { return Holovibes::instance().get_cd().composite_p_activated_v; }
inline void set_composite_p_activated_v(bool composite_p_activated_v)
{
    Holovibes::instance().get_cd().composite_p_activated_v = composite_p_activated_v;
}
inline bool get_composite_auto_weights_() { return Holovibes::instance().get_cd().composite_auto_weights_; }
inline void set_composite_auto_weights_(bool composite_auto_weights_)
{
    Holovibes::instance().get_cd().composite_auto_weights_ = composite_auto_weights_;
}

inline bool get_fast_pipe() { return Holovibes::instance().get_cd().fast_pipe; }
inline void set_fast_pipe(bool fast_pipe) { Holovibes::instance().get_cd().fast_pipe = fast_pipe; }

inline uint get_start_frame() { return Holovibes::instance().get_cd().start_frame; }
inline void set_start_frame(uint start_frame) { Holovibes::instance().get_cd().start_frame = start_frame; }
inline uint get_end_frame() { return Holovibes::instance().get_cd().end_frame; }
inline void set_end_frame(uint end_frame) { Holovibes::instance().get_cd().end_frame = end_frame; }

#pragma endregion

} // namespace holovibes::api