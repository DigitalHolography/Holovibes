#pragma once

#include "API.hh"

namespace holovibes::api
{
inline ComputeDescriptor& get_cd() { return Holovibes::instance().get_cd(); }

inline Computation get_compute_mode() { return get_cd().get_compute_mode(); }
inline void set_compute_mode(Computation mode) { return get_cd().set_compute_mode(mode); }

inline SpaceTransformation get_space_transformation() { return GSH::instance().get_space_transformation(); }

inline TimeTransformation get_time_transformation() { return GSH::instance().get_time_transformation(); }

inline ImgType get_img_type() { return GSH::instance().get_img_type(); }
inline void set_img_type(ImgType type) { return GSH::instance().set_img_type(type); }

inline WindowKind get_current_window_type() { return GSH::instance().get_current_window_type(); }

inline uint get_batch_size() { return GSH::instance().get_batch_size(); }
inline void set_batch_size(uint value) { GSH::instance().set_batch_size(value); }

inline uint get_time_transformation_stride() { return GSH::instance().get_time_transformation_stride(); }
inline void set_time_transformation_stride(uint value) { GSH::instance().set_time_transformation_stride(value); }

inline uint get_time_transformation_size() { return GSH::instance().get_time_transformation_size(); }
inline void set_time_transformation_size(uint value) { GSH::instance().set_time_transformation_size(value); }

inline float get_lambda() { return GSH::instance().get_lambda(); }
inline void set_lambda(float value) { GSH::instance().set_lambda(value); }

inline float get_z_distance() { return GSH::instance().get_z_distance(); }

inline float get_contrast_lower_threshold() { return get_cd().get_contrast_lower_threshold(); }
inline void set_contrast_lower_threshold(float value) { get_cd().set_contrast_lower_threshold(value); }

inline float get_contrast_upper_threshold() { return get_cd().get_contrast_upper_threshold(); }
inline void set_contrast_upper_threshold(float value) { get_cd().set_contrast_upper_threshold(value); }

inline uint get_cuts_contrast_p_offset() { return get_cd().get_cuts_contrast_p_offset(); }
inline void set_cuts_contrast_p_offset(uint value) { get_cd().set_cuts_contrast_p_offset(value); }

inline float get_pixel_size() { return get_cd().get_pixel_size(); }

inline unsigned get_renorm_constant() { return get_cd().get_renorm_constant(); }
inline void set_renorm_constant(unsigned int value) { get_cd().set_renorm_constant(value); }

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

inline int get_filter2d_smooth_low() { return get_cd().get_filter2d_smooth_low(); }
inline void set_filter2d_smooth_low(int value) { get_cd().set_filter2d_smooth_low(value); }

inline int get_filter2d_smooth_high() { return get_cd().get_filter2d_smooth_high(); }
inline void set_filter2d_smooth_high(int value) { get_cd().set_filter2d_smooth_high(value); }

inline float get_display_rate() { return get_cd().get_display_rate(); }
inline void set_display_rate(float value) { get_cd().set_display_rate(value); }

inline View_XY get_x(void) { return GSH::instance().get_x(); }
inline View_XY get_y(void) { return GSH::instance().get_y(); }

inline uint get_img_accu_xy_level() { return GSH::instance().get_xy_img_accu_level(); }

inline uint get_img_accu_xz_level() { return GSH::instance().get_xz_img_accu_level(); }

inline uint get_img_accu_yz_level() { return GSH::instance().get_yz_img_accu_level(); }

inline uint get_p_index() { return GSH::instance().get_p_index(); }

inline View_PQ get_p(void) { return GSH::instance().get_p(); }
inline View_PQ get_q(void) { return GSH::instance().get_q(); }

inline int get_p_accu_level() { return GSH::instance().get_p_accu_level(); }

inline uint get_x_cuts() { return GSH::instance().get_x_cuts(); }

inline int get_x_accu_level() { return GSH::instance().get_x_accu_level(); }

inline uint get_y_cuts() { return GSH::instance().get_y_cuts(); }

inline int get_y_accu_level() { return GSH::instance().get_y_accu_level(); }

inline uint get_q_index() { return GSH::instance().get_q_index(); }

inline uint get_q_accu_level() { return GSH::instance().get_q_accu_level(); }

inline float get_reticle_scale() { return get_cd().get_reticle_scale(); }

inline uint get_raw_bitshift() { return get_cd().get_raw_bitshift(); }

inline CompositeKind get_composite_kind() { return get_cd().get_composite_kind(); }

inline bool get_flip_enabled() { return GSH::instance().get_flip_enabled(); }

inline double get_rotation() { return GSH::instance().get_rotation(); }

// RGB
inline uint get_composite_p_red() { return get_cd().get_rgb_p_min(); }

inline uint get_composite_p_blue() { return get_cd().get_rgb_p_max(); }

inline float get_weight_r() { return get_cd().get_weight_r(); }

inline float get_weight_g() { return get_cd().get_weight_g(); }

inline float get_weight_b() { return get_cd().get_weight_b(); }

// HSV
inline uint get_composite_p_min_h() { return get_cd().get_composite_p_min_h(); }

inline uint get_composite_p_max_h() { return get_cd().get_composite_p_max_h(); }

inline float get_slider_h_threshold_min() { return get_cd().get_slider_h_threshold_min(); }
inline void set_slider_h_threshold_min(float value) { get_cd().set_slider_h_threshold_min(value); }

inline float get_slider_h_threshold_max() { return get_cd().get_slider_h_threshold_max(); }
inline void set_slider_h_threshold_max(float value) { get_cd().set_slider_h_threshold_max(value); }

inline float get_composite_low_h_threshold() { return get_cd().get_composite_low_h_threshold(); }

inline float get_composite_high_h_threshold() { return get_cd().get_composite_high_h_threshold(); }

inline uint get_h_blur_kernel_size() { return get_cd().get_h_blur_kernel_size(); }

inline uint get_composite_p_min_s() { return get_cd().get_composite_p_min_s(); }
inline uint get_composite_p_max_s() { return get_cd().get_composite_p_max_s(); }

inline float get_slider_s_threshold_min() { return get_cd().get_slider_s_threshold_min(); }
inline void set_slider_s_threshold_min(float value) { get_cd().set_slider_s_threshold_min(value); }

inline float get_slider_s_threshold_max() { return get_cd().get_slider_s_threshold_max(); }
inline void set_slider_s_threshold_max(float value) { get_cd().set_slider_s_threshold_max(value); }

inline float get_composite_low_s_threshold() { return get_cd().get_composite_low_s_threshold(); }

inline float get_composite_high_s_threshold() { return get_cd().get_composite_high_s_threshold(); }

inline uint get_composite_p_min_v() { return get_cd().get_composite_p_min_v(); }

inline uint get_composite_p_max_v() { return get_cd().get_composite_p_max_v(); }

inline float get_slider_v_threshold_min() { return get_cd().get_slider_v_threshold_min(); }
inline void set_slider_v_threshold_min(float value) { get_cd().set_slider_v_threshold_min(value); }

inline float get_slider_v_threshold_max() { return get_cd().get_slider_v_threshold_max(); }
inline void set_slider_v_threshold_max(float value) { get_cd().set_slider_v_threshold_max(value); }

inline float get_composite_low_v_threshold() { return get_cd().get_composite_low_v_threshold(); }

inline float get_composite_high_v_threshold() { return get_cd().get_composite_high_v_threshold(); }

inline int get_unwrap_history_size() { return get_cd().get_unwrap_history_size(); }

inline bool get_is_computation_stopped() { return get_cd().get_is_computation_stopped(); }

inline bool get_convolution_enabled() { return GSH::instance().get_convolution_enabled(); }
inline void set_convolution_enabled(bool value) { GSH::instance().set_convolution_enabled(value); }

inline bool get_divide_convolution_enabled() { return get_cd().get_divide_convolution_enabled(); }

inline bool get_renorm_enabled() { return get_cd().get_renorm_enabled(); }

inline bool get_fft_shift_enabled() { return get_cd().get_fft_shift_enabled(); }

inline bool get_log_scale_slice_xy_enabled() { return GSH::instance().get_xy_log_scale_slice_enabled(); }

inline bool get_log_scale_slice_xz_enabled() { return GSH::instance().get_xz_log_scale_slice_enabled(); }

inline bool get_log_scale_slice_yz_enabled() { return GSH::instance().get_yz_log_scale_slice_enabled(); }

inline bool get_contrast_enabled() { return GSH::instance().get_contrast_enabled(); }

inline bool get_contrast_auto_refresh() { return GSH::instance().get_contrast_auto_refresh(); }

inline bool get_contrast_invert() { return GSH::instance().get_contrast_invert(); }

inline bool get_filter2d_enabled() { return get_cd().get_filter2d_enabled(); }

inline bool get_filter2d_view_enabled() { return get_cd().get_filter2d_view_enabled(); }

inline bool get_3d_cuts_view_enabled() { return get_cd().get_3d_cuts_view_enabled(); }
inline void set_3d_cuts_view_enabled(bool value) { get_cd().set_3d_cuts_view_enabled(value); }

inline bool get_lens_view_enabled() { return get_cd().get_lens_view_enabled(); }

inline bool get_chart_display_enabled() { return get_cd().get_chart_display_enabled(); }
inline bool get_chart_record_enabled() { return get_cd().get_chart_record_enabled(); }

inline bool get_raw_view_enabled() { return get_cd().get_raw_view_enabled(); }

inline bool get_synchronized_record() { return get_cd().get_synchronized_record(); }

inline bool get_reticle_display_enabled() { return get_cd().get_reticle_display_enabled(); }

inline bool get_h_blur_activated() { return get_cd().get_h_blur_activated(); }

inline bool get_composite_p_activated_s() { return get_cd().get_composite_p_activated_s(); }

inline bool get_composite_p_activated_v() { return get_cd().get_composite_p_activated_v(); }

inline bool get_composite_auto_weights() { return get_cd().get_composite_auto_weights(); }

inline uint get_start_frame() { return get_cd().get_start_frame(); }

inline uint get_end_frame() { return get_cd().get_end_frame(); }

inline uint get_file_buffer_size() { return get_cd().get_file_buffer_size(); }
inline void set_file_buffer_size(uint value) { get_cd().set_file_buffer_size(value); }

inline uint get_input_buffer_size() { return get_cd().get_input_buffer_size(); }
inline void set_input_buffer_size(uint value) { get_cd().set_input_buffer_size(value); }

inline uint get_output_buffer_size() { return get_cd().get_output_buffer_size(); }
inline void set_output_buffer_size(uint value) { get_cd().set_output_buffer_size(value); }

inline uint get_record_buffer_size() { return get_cd().get_record_buffer_size(); }
inline void set_record_buffer_size(uint value) { get_cd().set_record_buffer_size(value); }

inline uint get_time_transformation_cuts_output_buffer_size()
{
    return get_cd().get_time_transformation_cuts_output_buffer_size();
}
inline void set_time_transformation_cuts_output_buffer_size(uint value)
{
    get_cd().set_time_transformation_cuts_output_buffer_size(value);
}

inline const camera::FrameDescriptor& get_fd() { return Holovibes::instance().get_gpu_input_queue()->get_fd(); };

inline std::shared_ptr<Pipe> get_compute_pipe() { return Holovibes::instance().get_compute_pipe(); };

inline std::shared_ptr<Queue> get_gpu_output_queue() { return Holovibes::instance().get_gpu_output_queue(); };

inline std::shared_ptr<BatchInputQueue> get_gpu_input_queue() { return Holovibes::instance().get_gpu_input_queue(); };

#pragma endregion

} // namespace holovibes::api
