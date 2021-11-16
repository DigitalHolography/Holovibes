#pragma once

#include "API.hh"

namespace holovibes::api
{

inline ComputeDescriptor& get_cd() { return Holovibes::instance().get_cd(); }

inline Computation get_compute_mode() { return get_cd().get_compute_mode(); }
inline void set_compute_mode(Computation mode) { return get_cd().set_compute_mode(mode); }

inline SpaceTransformation get_space_transformation() { return get_cd().get_space_transformation(); }

inline TimeTransformation get_time_transformation() { return get_cd().get_time_transformation(); }

inline ImgType get_img_type() { return get_cd().get_img_type(); }
inline void set_img_type(ImgType type) { return get_cd().set_img_type(type); }

inline WindowKind get_current_window() { return get_cd().get_current_window(); }

inline uint get_batch_size() { return GSH::instance().batch_query().value; }
inline void set_batch_size(uint value) { GSH::instance().batch_command({value}); }

inline uint get_time_transformation_stride() { return get_cd().get_time_transformation_stride(); }
inline void set_time_transformation_stride(uint value) { get_cd().set_time_transformation_stride(value); }

inline uint get_time_transformation_size() { return get_cd().get_time_transformation_size(); }
inline void set_time_transformation_size(uint value) { get_cd().set_time_transformation_size(value); }

inline float get_lambda() { return get_cd().get_lambda(); }
inline float get_zdistance() { return get_cd().get_zdistance(); }

inline float get_contrast_lower_threshold() { return get_cd().get_contrast_lower_threshold(); }

inline float get_contrast_upper_threshold() { return get_cd().get_contrast_upper_threshold(); }

inline uint get_cuts_contrast_p_offset() { return get_cd().get_cuts_contrast_p_offset(); }

inline float get_pixel_size() { return get_cd().get_pixel_size(); }

inline unsigned get_renorm_constant() { return get_cd().get_renorm_constant(); }

inline int get_filter2d_n1() { return get_cd().get_filter2d_n1(); }
inline void set_filter2d_n1(int value) { get_cd().set_filter2d_n1(value); }

inline int get_filter2d_n2() { return get_cd().get_filter2d_n2(); }
inline void set_filter2d_n2(int value) { get_cd().set_filter2d_n2(value); }

inline int get_filter2d_smooth_low() { return get_cd().get_filter2d_smooth_low(); }

inline int get_filter2d_smooth_high() { return get_cd().get_filter2d_smooth_high(); }

inline float get_display_rate() { return get_cd().get_display_rate(); }

inline uint get_img_acc_slice_xy_level() { return get_cd().get_img_accu_slice_xy_level(); }

inline uint get_img_acc_slice_xz_level() { return get_cd().get_img_acc_slice_xz_level(); }

inline uint get_img_acc_slice_yz_level() { return get_cd().get_img_acc_slice_yz_level(); }

inline uint get_pindex() { return get_cd().get_p_index(); }

inline int get_p_acc_level() { return get_cd().get_p_acc_level(); }

inline uint get_x_cuts() { return get_cd().get_x_cuts(); }

inline int get_x_acc_level() { return get_cd().get_x_acc_level(); }

inline uint get_y_cuts() { return get_cd().get_y_cuts(); }

inline int get_y_acc_level() { return get_cd().get_y_acc_level(); }

inline uint get_q_index() { return get_cd().get_q_index(); }

inline uint get_q_acc_level() { return get_cd().get_q_acc_level(); }

inline float get_reticle_scale() { return get_cd().get_reticle_scale(); }

inline uint get_raw_bitshift() { return get_cd().get_raw_bitshift(); }

inline CompositeKind get_composite_kind() { return get_cd().get_composite_kind(); }

inline bool get_flip_enabled() { return get_cd().get_flip_enabled(); }

inline double get_rotation() { return get_cd().get_rotation(); }

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

inline bool get_convolution_enabled() { return get_cd().get_convolution_enabled(); }

inline bool get_divide_convolution_enabled() { return get_cd().get_divide_convolution_enabled(); }

inline bool get_renorm_enabled() { return get_cd().get_renorm_enabled(); }

inline bool get_fft_shift_enabled() { return get_cd().get_fft_shift_enabled(); }

inline bool get_frame_record_enabled() { return get_cd().get_frame_record_enabled(); }

inline bool get_log_scale_slice_xy_enabled() { return get_cd().get_log_scale_slice_xy_enabled(); }

inline bool get_log_scale_slice_xz_enabled() { return get_cd().get_log_scale_slice_xz_enabled(); }

inline bool get_log_scale_slice_yz_enabled() { return get_cd().get_log_scale_slice_yz_enabled(); }

inline bool get_log_scale_filter2d_enabled() { return get_cd().get_log_scale_filter2d_enabled(); }

inline bool get_contrast_enabled() { return get_cd().get_contrast_enabled(); }

inline bool get_contrast_auto_refresh() { return get_cd().get_contrast_auto_refresh(); }

inline bool get_contrast_invert() { return get_cd().get_contrast_invert(); }

inline bool get_filter2d_enabled() { return get_cd().get_filter2d_enabled(); }

inline bool get_filter2d_view_enabled() { return get_cd().get_filter2d_view_enabled(); }

inline bool get_time_transformation_cuts_enabled() { return get_cd().get_time_transformation_cuts_enabled(); }
inline void set_time_transformation_cuts_enabled(bool value) { get_cd().set_time_transformation_cuts_enabled(value); }

inline bool get_lens_view_enabled() { return get_cd().get_lens_view_enabled(); }
inline void set_lens_view_enabled(bool value) { return get_cd().set_lens_view_enabled(value); }

inline bool get_chart_display_enabled() { return get_cd().get_chart_display_enabled(); }
inline bool get_chart_record_enabled() { return get_cd().get_chart_record_enabled(); }

inline bool get_img_acc_slice_xy_enabled() { return get_cd().get_img_acc_slice_xy_enabled(); }

inline bool get_img_acc_slice_xz_enabled() { return get_cd().get_img_acc_slice_xz_enabled(); }

inline bool get_img_acc_slice_yz_enabled() { return get_cd().get_img_acc_slice_yz_enabled(); }

inline bool get_p_accu_enabled() { return get_cd().get_p_accu_enabled(); }

inline bool get_x_accu_enabled() { return get_cd().get_x_accu_enabled(); }

inline bool get_y_accu_enabled() { return get_cd().get_y_accu_enabled(); }

inline bool get_q_acc_enabled() { return get_cd().get_q_acc_enabled(); }

inline bool get_raw_view_enabled() { return get_cd().get_raw_view_enabled(); }

inline bool get_synchronized_record() { return get_cd().get_synchronized_record(); }

inline bool get_reticle_view_enabled() { return get_cd().get_reticle_view_enabled(); }

inline bool get_h_blur_activated() { return get_cd().get_h_blur_activated(); }

inline bool get_composite_p_activated_s() { return get_cd().get_composite_p_activated_s(); }

inline bool get_composite_p_activated_v() { return get_cd().get_composite_p_activated_v(); }

inline bool get_composite_auto_weights() { return get_cd().get_composite_auto_weights(); }

inline uint get_start_frame() { return get_cd().get_start_frame(); }

inline uint get_end_frame() { return get_cd().get_end_frame(); }

inline uint get_input_buffer_size() { return get_cd().get_input_buffer_size(); }
inline uint get_output_buffer_size() { return get_cd().get_output_buffer_size(); }

inline const camera::FrameDescriptor& get_fd() { return Holovibes::instance().get_gpu_input_queue()->get_fd(); };

inline std::shared_ptr<Pipe> get_compute_pipe() { return Holovibes::instance().get_compute_pipe(); };

inline std::shared_ptr<Queue> get_gpu_output_queue() { return Holovibes::instance().get_gpu_output_queue(); };

inline std::shared_ptr<BatchInputQueue> get_gpu_input_queue() { return Holovibes::instance().get_gpu_input_queue(); };

#pragma endregion

} // namespace holovibes::api
