#pragma once

#include "API.hh"

namespace holovibes::api
{
inline ImgType get_img_type() { return GSH::instance().get_img_type(); }
inline void set_img_type(ImgType type) { return GSH::instance().set_img_type(type); }

inline WindowKind get_current_window_type() { return GSH::instance().get_current_window_type(); }

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

inline View_XY get_x(void) { return GSH::instance().get_x(); }
inline View_XY get_y(void) { return GSH::instance().get_y(); }

inline uint get_img_accu_xy_level() { return GSH::instance().get_xy_img_accu_level(); }
inline void set_img_accu_xy_level(uint value) { GSH::instance().set_xy_img_accu_level(value); }

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

inline float get_reticle_scale() { return GSH::instance().get_reticle_scale(); }
inline void set_reticle_scale(float value) { GSH::instance().set_reticle_scale(value); }

inline bool get_flip_enabled() { return GSH::instance().get_flip_enabled(); }

inline double get_rotation() { return GSH::instance().get_rotation(); }

// HSV
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

inline uint get_start_frame() { return GSH::instance().get_start_frame(); }
inline void set_start_frame(uint value) { GSH::instance().set_start_frame(value); }

inline uint get_end_frame() { return GSH::instance().get_end_frame(); }
inline void set_end_frame(uint value) { GSH::instance().set_end_frame(value); }

inline uint get_file_buffer_size() { return GSH::instance().get_file_buffer_size(); }
inline void set_file_buffer_size(uint value) { GSH::instance().set_file_buffer_size(value); }

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
