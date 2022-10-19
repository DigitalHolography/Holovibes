#pragma once

#include <mutex>

#include "fast_updates_holder.hh"
#include "caches.hh"
#include "entities.hh"
#include "view_struct.hh"
#include "rendering_struct.hh"
#include "composite_struct.hh"
#include "internals_struct.hh"
#include "advanced_struct.hh"

namespace holovibes
{

using entities::Span;

/*! \class GSH
 *
 * \brief The GSH (global state holder), is where all the global state of the program is stored.
 *
 * Its goal is to register changes commanded by the API (which comes from user events), dispatch the changes
 * to each worker, and provide informations queried by the API. It relies on several structures and mecanisms :
 *
 * Queries and Commands : every data needed by the API shall be obtained in the form of structured Queries, provided by
 * the GSH. This guarantees that only needed data is accessed to, and grants better code readability.
 * The same principle is applied to changes comming from the API, which come in the form of structured Commands.
 *
 * MicroCache : local state holder belonging to a worker. Previously, each worker had to fetch data at the same
 * place ; therefore, all variables had to be atomic, with the aim to be thread safe. Furthermore, since global state
 * was used in the pipe, directly modifying the state was often not possible (changing operations or variables which
 * have impact on buffers'size would have caused incoherent computations and/or segfaults and undefined behaviors). The
 * ComputeWorker now possess MicroCaches, containing all the state it needs. Those MicroCaches are accessed only by
 * their worker, and are synchronized with the GSH when each worker chooses to (using a trigger system). The
 * implementation of MicroCaches enabled the direct modification of the state, since the state used in the pipe is now
 * desynchronized from the GSH.
 * More informations and explanations concerning their synchronization with the GSH are provided in files micro-cache.hh
 * and micro-cache.hxx.
 *
 * FastUpdateHolder : the fastUpdateHolder is a templated map which is used by the informationWorker to access and
 * display information (like fps and queue occupancy) at a high rate, since this needs to be updated continuously.
 */
class GSH
{
    static GSH* instance_;

  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    // static inline GSH& instance() { return *instance_; }
    static GSH& instance();

    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

#pragma region(collapsed) GETTERS

    inline uint get_time_stride() const noexcept { return compute_cache_.get_time_stride(); }

    inline SpaceTransformation get_space_transformation() const noexcept
    {
        return compute_cache_.get_space_transformation();
    }

    inline TimeTransformation get_time_transformation() const noexcept
    {
        return compute_cache_.get_time_transformation();
    };

    inline uint get_batch_size() const noexcept { return compute_cache_.get_batch_size(); }
    inline uint get_time_transformation_size() const noexcept { return compute_cache_.get_time_transformation_size(); }
    inline float get_lambda() const noexcept { return compute_cache_.get_lambda(); }
    inline float get_z_distance() const noexcept { return compute_cache_.get_z_distance(); };
    inline bool get_convolution_enabled() const noexcept { return compute_cache_.get_convolution_enabled(); }
    inline const std::vector<float>& get_convo_matrix_const_ref()
    {
        return compute_cache_.get_convo_matrix_const_ref();
    };

    inline int get_filter2d_n1() const noexcept { return filter2d_cache_.get_filter2d_n1(); }
    inline int get_filter2d_n2() const noexcept { return filter2d_cache_.get_filter2d_n2(); }
    inline ImgType get_img_type() const noexcept { return view_cache_.get_img_type(); }

    inline ViewXY get_x() const noexcept { return view_cache_.get_x(); }
    inline int get_x_accu_level() const noexcept { return view_cache_.get_x().accu_level; }
    inline int get_x_cuts() const noexcept { return view_cache_.get_x().cuts; }

    inline ViewXY get_y() const noexcept { return view_cache_.get_y(); }
    inline int get_y_accu_level() const noexcept { return view_cache_.get_y().accu_level; }
    inline int get_y_cuts() const noexcept { return view_cache_.get_y().cuts; }

    inline ViewPQ get_p() const noexcept { return view_cache_.get_p(); }
    inline int get_p_accu_level() const noexcept { return view_cache_.get_p().accu_level; }
    inline uint get_p_index() const noexcept { return view_cache_.get_p().index; }

    inline ViewPQ get_q() const noexcept { return view_cache_.get_q(); }
    inline int get_q_accu_level() const noexcept { return view_cache_.get_q().accu_level; }
    inline uint get_q_index() const noexcept { return view_cache_.get_q().index; }

    inline ViewXYZ get_xy() const noexcept { return view_cache_.get_xy(); }
    inline bool get_xy_flip_enabled() const noexcept { return view_cache_.get_xy().flip_enabled; }
    inline float get_xy_rot() const noexcept { return view_cache_.get_xy().rot; }
    inline uint get_xy_img_accu_level() const noexcept { return view_cache_.get_xy().img_accu_level; }
    inline bool get_xy_log_scale_slice_enabled() const noexcept { return view_cache_.get_xy().log_enabled; }
    inline bool get_xy_contrast_enabled() const noexcept { return view_cache_.get_xy().contrast.enabled; }
    inline bool get_xy_contrast_auto_refresh() const noexcept { return view_cache_.get_xy().contrast.auto_refresh; }
    inline bool get_xy_contrast_invert() const noexcept { return view_cache_.get_xy().contrast.invert; }
    inline float get_xy_contrast_min() const noexcept { return view_cache_.get_xy().contrast.min; }
    inline float get_xy_contrast_max() const noexcept { return view_cache_.get_xy().contrast.max; }
    inline bool get_xy_img_accu_enabled() const noexcept { return view_cache_.get_xy().img_accu_level > 1; }

    inline ViewXYZ get_xz() const noexcept { return view_cache_.get_xz(); }
    inline bool get_xz_flip_enabled() const noexcept { return view_cache_.get_xz().flip_enabled; }
    inline float get_xz_rot() const noexcept { return view_cache_.get_xz().rot; }
    inline uint get_xz_img_accu_level() const noexcept { return view_cache_.get_xz().img_accu_level; }
    inline bool get_xz_log_scale_slice_enabled() const noexcept { return view_cache_.get_xz().log_enabled; }
    inline bool get_xz_contrast_enabled() const noexcept { return view_cache_.get_xz().contrast.enabled; }
    inline bool get_xz_contrast_auto_refresh() const noexcept { return view_cache_.get_xz().contrast.auto_refresh; }
    inline bool get_xz_contrast_invert() const noexcept { return view_cache_.get_xz().contrast.invert; }
    inline float get_xz_contrast_min() const noexcept { return view_cache_.get_xz().contrast.min; }
    inline float get_xz_contrast_max() const noexcept { return view_cache_.get_xz().contrast.max; }
    inline bool get_xz_img_accu_enabled() const noexcept { return view_cache_.get_xz().img_accu_level > 1; }

    inline ViewXYZ get_yz() const noexcept { return view_cache_.get_yz(); }
    inline bool get_yz_flip_enabled() const noexcept { return view_cache_.get_yz().flip_enabled; }
    inline float get_yz_rot() const noexcept { return view_cache_.get_yz().rot; }
    inline uint get_yz_img_accu_level() const noexcept { return view_cache_.get_yz().img_accu_level; }
    inline bool get_yz_log_scale_slice_enabled() const noexcept { return view_cache_.get_yz().log_enabled; }
    inline bool get_yz_contrast_enabled() const noexcept { return view_cache_.get_yz().contrast.enabled; }
    inline bool get_yz_contrast_auto_refresh() const noexcept { return view_cache_.get_yz().contrast.auto_refresh; }
    inline bool get_yz_contrast_invert() const noexcept { return view_cache_.get_yz().contrast.invert; }
    inline float get_yz_contrast_min() const noexcept { return view_cache_.get_yz().contrast.min; }
    inline float get_yz_contrast_max() const noexcept { return view_cache_.get_yz().contrast.max; }
    inline bool get_yz_img_accu_enabled() const noexcept { return view_cache_.get_yz().img_accu_level > 1; }

    inline ViewWindow get_filter2d() const noexcept { return view_cache_.get_filter2d(); }
    inline bool get_filter2d_contrast_enabled() const noexcept { return view_cache_.get_filter2d().contrast.enabled; }
    inline bool get_filter2d_contrast_invert() const noexcept { return view_cache_.get_filter2d().contrast.invert; }
    inline float get_filter2d_contrast_min() const noexcept { return view_cache_.get_filter2d().contrast.min; }
    inline float get_filter2d_contrast_max() const noexcept { return view_cache_.get_filter2d().contrast.max; }
    inline bool get_filter2d_log_scale_slice_enabled() const noexcept { return view_cache_.get_filter2d().log_enabled; }
    inline bool get_filter2d_contrast_auto_refresh() const noexcept
    {
        return view_cache_.get_filter2d().contrast.auto_refresh;
    }

    inline WindowKind get_current_window_type() const noexcept { return view_cache_.get_current_window(); }

    inline bool get_contrast_auto_refresh() const noexcept { return get_current_window().contrast.auto_refresh; }
    inline bool get_contrast_invert() const noexcept { return get_current_window().contrast.invert; }
    inline bool get_contrast_enabled() const noexcept { return get_current_window().contrast.enabled; }

    bool is_current_window_xyz_type() const;

    // Over current window
    float get_contrast_min() const;
    float get_contrast_max() const;
    double get_rotation() const;
    bool get_flip_enabled() const;
    bool get_img_log_scale_slice_enabled() const;
    unsigned get_img_accu_level() const;

    inline bool get_divide_convolution_enabled() const { return compute_cache_.get_divide_convolution_enabled(); };

    inline bool get_lens_view_enabled() const { return view_cache_.get_lens_view_enabled(); };

    inline uint get_input_fps() const { return compute_cache_.get_input_fps(); };

    inline bool get_frame_record_enabled() const { return export_cache_.get_frame_record_enabled(); };

    inline bool get_chart_display_enabled() const { return view_cache_.get_chart_display_enabled(); };

    inline bool get_chart_record_enabled() const { return export_cache_.get_chart_record_enabled(); };

    inline Computation get_compute_mode() const noexcept { return compute_cache_.get_compute_mode(); };

    inline bool get_filter2d_enabled() const noexcept { return view_cache_.get_filter2d_enabled(); }

    inline bool get_filter2d_view_enabled() const noexcept { return view_cache_.get_filter2d_view_enabled(); }

    inline CompositeKind get_composite_kind() const noexcept { return composite_cache_.get_composite_kind(); }

    inline bool get_fft_shift_enabled() const noexcept { return view_cache_.get_fft_shift_enabled(); }

    inline bool get_raw_view_enabled() const noexcept { return view_cache_.get_raw_view_enabled(); }

    inline bool get_composite_auto_weights() const noexcept { return composite_cache_.get_composite_auto_weights(); }

    inline uint get_start_frame() const noexcept { return import_cache_.get_start_frame(); }
    inline uint get_end_frame() const noexcept { return import_cache_.get_end_frame(); }

    inline float get_display_rate() const noexcept { return advanced_cache_.get_display_rate(); }

    inline bool get_cuts_view_enabled() const noexcept { return view_cache_.get_cuts_view_enabled(); }

    inline uint get_file_buffer_size() const noexcept { return file_read_cache_.get_file_buffer_size(); }

    inline uint get_input_buffer_size() const noexcept { return advanced_cache_.get_input_buffer_size(); }

    inline uint get_record_buffer_size() const noexcept { return advanced_cache_.get_record_buffer_size(); }

    inline uint get_output_buffer_size() const noexcept { return advanced_cache_.get_output_buffer_size(); }

    inline float get_pixel_size() const noexcept { return compute_cache_.get_pixel_size(); }

    inline uint get_unwrap_history_size() const noexcept { return compute_cache_.get_unwrap_history_size(); }

    inline bool get_is_computation_stopped() const noexcept { return compute_cache_.get_is_computation_stopped(); }

    inline bool get_renorm_enabled() const noexcept { return view_cache_.get_renorm_enabled(); }

    // RGB
    inline CompositeRGB get_rgb() const noexcept { return composite_cache_.get_rgb(); }
    inline uint get_rgb_p_min() const noexcept { return composite_cache_.get_rgb().p.min; }
    inline uint get_rgb_p_max() const noexcept { return composite_cache_.get_rgb().p.max; }
    inline float get_weight_r() const noexcept { return composite_cache_.get_rgb().weight.r; }
    inline float get_weight_g() const noexcept { return composite_cache_.get_rgb().weight.g; }
    inline float get_weight_b() const noexcept { return composite_cache_.get_rgb().weight.b; }

    // HSV
    inline CompositeHSV get_hsv() const noexcept { return composite_cache_.get_hsv(); }
    inline uint get_composite_p_min_h() const noexcept { return composite_cache_.get_hsv().h.p.min; }
    inline uint get_composite_p_max_h() const noexcept { return composite_cache_.get_hsv().h.p.max; }

    inline float get_slider_h_threshold_min() const noexcept
    {
        return composite_cache_.get_hsv().h.slider_threshold.min;
    }
    inline float get_slider_h_threshold_max() const noexcept
    {
        return composite_cache_.get_hsv().h.slider_threshold.max;
    }
    inline unsigned int get_raw_bitshift() const noexcept { return advanced_cache_.get_raw_bitshift(); }

    inline float get_composite_low_h_threshold() const noexcept { return composite_cache_.get_hsv().h.threshold.min; }
    inline float get_composite_high_h_threshold() const noexcept { return composite_cache_.get_hsv().h.threshold.max; }
    inline uint get_h_blur_kernel_size() const noexcept { return composite_cache_.get_hsv().h.blur.kernel_size; }
    inline uint get_composite_p_min_s() const noexcept { return composite_cache_.get_hsv().s.p.min; }
    inline uint get_composite_p_max_s() const noexcept { return composite_cache_.get_hsv().s.p.max; }
    inline float get_slider_s_threshold_min() const noexcept
    {
        return composite_cache_.get_hsv().s.slider_threshold.min;
    }
    inline float get_slider_s_threshold_max() const noexcept
    {
        return composite_cache_.get_hsv().s.slider_threshold.max;
    }
    inline float get_composite_low_s_threshold() const noexcept { return composite_cache_.get_hsv().s.threshold.min; }
    inline float get_composite_high_s_threshold() const noexcept { return composite_cache_.get_hsv().s.threshold.max; }
    inline uint get_composite_p_min_v() const noexcept { return composite_cache_.get_hsv().v.p.min; }
    inline uint get_composite_p_max_v() const noexcept { return composite_cache_.get_hsv().v.p.max; }
    inline float get_slider_v_threshold_min() const noexcept
    {
        return composite_cache_.get_hsv().v.slider_threshold.min;
    }
    inline float get_slider_v_threshold_max() const noexcept
    {
        return composite_cache_.get_hsv().v.slider_threshold.max;
    }
    inline float get_composite_low_v_threshold() const noexcept { return composite_cache_.get_hsv().v.threshold.min; }
    inline float get_composite_high_v_threshold() const noexcept { return composite_cache_.get_hsv().v.threshold.max; }
    inline bool get_h_blur_activated() const noexcept { return composite_cache_.get_hsv().h.blur.enabled; }
    inline bool get_composite_p_activated_s() const noexcept { return composite_cache_.get_hsv().s.p.activated; }
    inline bool get_composite_p_activated_v() const noexcept { return composite_cache_.get_hsv().v.p.activated; }

    inline float get_reticle_scale() const noexcept { return view_cache_.get_reticle_scale(); }

    inline uint get_time_transformation_cuts_output_buffer_size() const noexcept
    {
        return compute_cache_.get_time_transformation_cuts_output_buffer_size();
    }

    inline int get_filter2d_smooth_low() const noexcept { return filter2d_cache_.get_filter2d_smooth_low(); }

    inline int get_filter2d_smooth_high() const noexcept { return filter2d_cache_.get_filter2d_smooth_high(); }

    inline float get_contrast_lower_threshold() const noexcept
    {
        return advanced_cache_.get_contrast_lower_threshold();
    }

    inline float get_contrast_upper_threshold() const noexcept
    {
        return advanced_cache_.get_contrast_upper_threshold();
    }

    inline unsigned get_renorm_constant() const noexcept { return advanced_cache_.get_renorm_constant(); }

    inline uint get_cuts_contrast_p_offset() const noexcept { return advanced_cache_.get_cuts_contrast_p_offset(); }

    inline bool get_reticle_display_enabled() const noexcept { return view_cache_.get_reticle_display_enabled(); }

    inline units::RectFd get_signal_zone() const noexcept { return zone_cache_.get_signal_zone(); }
    inline units::RectFd get_noise_zone() const noexcept { return zone_cache_.get_noise_zone(); }
    inline units::RectFd get_composite_zone() const noexcept { return zone_cache_.get_composite_zone(); }
    inline units::RectFd get_zoomed_zone() const noexcept { return zone_cache_.get_zoomed_zone(); }
    inline units::RectFd get_reticle_zone() const noexcept { return zone_cache_.get_reticle_zone(); }

#pragma endregion

#pragma region(collapsed) SETTERS
    void set_batch_size(uint value);
    void set_time_transformation_size(uint value);
    void set_time_stride(uint value);
    void disable_convolution();
    void enable_convolution(std::optional<std::string> file);
    void set_convolution_enabled(bool value);

    inline void set_space_transformation(const SpaceTransformation value) noexcept
    {
        compute_cache_.set_space_transformation(value);
    }

    inline void set_time_transformation(const TimeTransformation value) noexcept
    {
        compute_cache_.set_time_transformation(value);
    }

    inline void set_lambda(float value) noexcept { compute_cache_.set_lambda(value); }

    inline void set_z_distance(float value) noexcept { compute_cache_.set_z_distance(value); }

    inline void set_filter2d_n1(int value) noexcept { filter2d_cache_.set_filter2d_n1(value); }
    inline void set_filter2d_n2(int value) noexcept { filter2d_cache_.set_filter2d_n2(value); }

    inline void set_img_type(ImgType value) noexcept { view_cache_.set_img_type(value); }

    inline void set_x(ViewXY value) noexcept { view_cache_.set_x(value); }
    inline void set_x_accu_level(int value) noexcept { view_cache_.get_x_ref()->accu_level = value; }
    inline void set_x_cuts(int value) noexcept { view_cache_.get_x_ref()->cuts = value; }

    inline void set_y(ViewXY value) noexcept { view_cache_.set_y(value); }
    inline void set_y_accu_level(int value) noexcept { view_cache_.get_y_ref()->accu_level = value; }
    inline void set_y_cuts(int value) noexcept { view_cache_.get_y_ref()->cuts = value; }

    inline void set_p(ViewPQ value) noexcept { view_cache_.set_p(value); }
    inline void set_p_accu_level(int value) noexcept { view_cache_.get_p_ref()->accu_level = value; }
    inline void set_p_index(uint value) noexcept
    {
        view_cache_.get_p_ref()->index = value;
        notify_callback_();
    }

    inline void set_q(ViewPQ value) noexcept { view_cache_.set_q(value); }
    inline void set_q_accu_level(int value) noexcept { view_cache_.get_q_ref()->accu_level = value; }
    inline void set_q_index(uint value) noexcept { view_cache_.get_q_ref()->index = value; }

    inline void set_xy(ViewXYZ value) noexcept { view_cache_.set_xy(value); }
    inline void set_xy_flip_enabled(bool value) noexcept { view_cache_.get_xy_ref()->flip_enabled = value; }
    inline void set_xy_rot(float value) noexcept { view_cache_.get_xy_ref()->rot = value; }
    inline void set_xy_img_accu_level(uint value) noexcept { view_cache_.get_xy_ref()->img_accu_level = value; }
    inline void set_xy_log_scale_slice_enabled(bool value) noexcept { view_cache_.get_xy_ref()->log_enabled = value; }
    inline void set_xy_contrast_enabled(bool value) noexcept { view_cache_.get_xy_ref()->contrast.enabled = value; }
    inline void set_xy_contrast_auto_refresh(bool value) noexcept
    {
        view_cache_.get_xy_ref()->contrast.auto_refresh = value;
    }
    inline void set_xy_contrast_invert(bool value) noexcept { view_cache_.get_xy_ref()->contrast.invert = value; }
    inline void set_xy_contrast_min(float value) noexcept
    {
        view_cache_.get_xy_ref()->contrast.min = value > 1.0f ? value : 1.0f;
    }
    inline void set_xy_contrast_max(float value) noexcept
    {
        view_cache_.get_xy_ref()->contrast.max = value > 1.0f ? value : 1.0f;
    }

    inline void set_xz(ViewXYZ value) noexcept { view_cache_.set_xz(value); }
    inline void set_xz_flip_enabled(bool value) noexcept { view_cache_.get_xz_ref()->flip_enabled = value; }
    inline void set_xz_rot(float value) noexcept { view_cache_.get_xz_ref()->rot = value; }
    inline void set_xz_img_accu_level(uint value) noexcept { view_cache_.get_xz_ref()->img_accu_level = value; }
    inline void set_xz_log_scale_slice_enabled(bool value) noexcept { view_cache_.get_xz_ref()->log_enabled = value; }
    inline void set_xz_contrast_enabled(bool value) noexcept { view_cache_.get_xz_ref()->contrast.enabled = value; }
    inline void set_xz_contrast_auto_refresh(bool value) noexcept
    {
        view_cache_.get_xz_ref()->contrast.auto_refresh = value;
    }
    inline void set_xz_contrast_invert(bool value) noexcept { view_cache_.get_xz_ref()->contrast.invert = value; }
    inline void set_xz_contrast_min(float value) noexcept
    {
        view_cache_.get_xz_ref()->contrast.min = value > 1.0f ? value : 1.0f;
    }
    inline void set_xz_contrast_max(float value) noexcept
    {
        view_cache_.get_xz_ref()->contrast.max = value > 1.0f ? value : 1.0f;
    }

    inline void set_yz(ViewXYZ value) noexcept { view_cache_.set_yz(value); }
    inline void set_yz_flip_enabled(bool value) noexcept { view_cache_.get_yz_ref()->flip_enabled = value; }
    inline void set_yz_rot(float value) noexcept { view_cache_.get_yz_ref()->rot = value; }
    inline void set_yz_img_accu_level(uint value) noexcept { view_cache_.get_yz_ref()->img_accu_level = value; }
    inline void set_yz_log_scale_slice_enabled(bool value) noexcept { view_cache_.get_yz_ref()->log_enabled = value; }
    inline void set_yz_contrast_enabled(bool value) noexcept { view_cache_.get_yz_ref()->contrast.enabled = value; }
    inline void set_yz_contrast_auto_refresh(bool value) noexcept
    {
        view_cache_.get_yz_ref()->contrast.auto_refresh = value;
    }
    inline void set_yz_contrast_invert(bool value) noexcept { view_cache_.get_yz_ref()->contrast.invert = value; }
    inline void set_yz_contrast_min(float value) noexcept
    {
        view_cache_.get_yz_ref()->contrast.min = value > 1.0f ? value : 1.0f;
    }
    inline void set_yz_contrast_max(float value) noexcept
    {
        view_cache_.get_yz_ref()->contrast.max = value > 1.0f ? value : 1.0f;
    }

    inline void set_filter2d(ViewWindow value) noexcept { view_cache_.set_filter2d(value); }
    inline void set_filter2d_log_scale_slice_enabled(bool value) noexcept
    {
        view_cache_.get_filter2d_ref()->log_enabled = value;
    }
    inline void set_filter2d_contrast_enabled(bool value) noexcept
    {
        view_cache_.get_filter2d_ref()->contrast.enabled = value;
    }
    inline void set_filter2d_contrast_auto_refresh(bool value) noexcept
    {
        view_cache_.get_filter2d_ref()->contrast.auto_refresh = value;
    }
    inline void set_filter2d_contrast_invert(bool value) noexcept
    {
        view_cache_.get_filter2d_ref()->contrast.invert = value;
    }
    inline void set_filter2d_contrast_min(float value) noexcept
    {
        view_cache_.get_filter2d_ref()->contrast.min = value > 1.0f ? value : 1.0f;
    }
    inline void set_filter2d_contrast_max(float value) noexcept
    {
        view_cache_.get_filter2d_ref()->contrast.max = value > 1.0f ? value : 1.0f;
    }

    inline void set_log_scale_filter2d_enabled(bool log_scale_filter2d_enabled) noexcept
    {
        view_cache_.get_filter2d_ref()->log_enabled = log_scale_filter2d_enabled;
    }

    // Over current window
    void set_contrast_enabled(bool contrast_enabled);
    void set_contrast_auto_refresh(bool contrast_auto_refresh);
    void set_contrast_invert(bool contrast_invert);
    void set_contrast_min(float value);
    void set_contrast_max(float value);
    void set_log_scale_slice_enabled(bool value);
    void set_accumulation_level(int value);
    void set_rotation(double value);
    void set_flip_enabled(double value);

    inline void set_divide_convolution_enabled(bool value) { compute_cache_.set_divide_convolution_enabled(value); };

    inline void set_lens_view_enabled(bool value) { view_cache_.set_lens_view_enabled(value); }

    inline void set_input_fps(uint value) { compute_cache_.set_input_fps(value); };

    inline void set_frame_record_enabled(bool value) { export_cache_.set_frame_record_enabled(value); }

    inline void set_chart_display_enabled(bool value) { view_cache_.set_chart_display_enabled(value); }

    inline void set_chart_record_enabled(bool value) { export_cache_.set_chart_record_enabled(value); }

    inline void set_compute_mode(Computation value) { compute_cache_.set_compute_mode(value); }

    inline void set_filter2d_enabled(bool value) { view_cache_.set_filter2d_enabled(value); }

    inline void set_filter2d_view_enabled(bool value) { view_cache_.set_filter2d_view_enabled(value); }

    inline void set_composite_kind(CompositeKind value) { composite_cache_.set_composite_kind(value); }

    void set_fft_shift_enabled(bool value);

    inline void set_raw_view_enabled(bool value) { view_cache_.set_raw_view_enabled(value); }

    inline void set_composite_auto_weights(bool value) { composite_cache_.set_composite_auto_weights(value); }

    inline void set_start_frame(uint value) { import_cache_.set_start_frame(value); }

    inline void set_end_frame(uint value) { import_cache_.set_end_frame(value); }

    inline void set_display_rate(float value) { advanced_cache_.set_display_rate(value); }

    inline void set_cuts_view_enabled(bool value) { view_cache_.set_cuts_view_enabled(value); }

    inline void set_file_buffer_size(uint value) { file_read_cache_.set_file_buffer_size(value); }

    inline void set_input_buffer_size(uint value) { advanced_cache_.set_input_buffer_size(value); }

    inline void set_record_buffer_size(uint value) { advanced_cache_.set_record_buffer_size(value); }

    inline void set_output_buffer_size(uint value) { advanced_cache_.set_output_buffer_size(value); }

    inline void set_pixel_size(float value)
    {
        std::lock_guard<std::mutex> lock(mutex_);
        compute_cache_.set_pixel_size(value);
    }

    inline void set_unwrap_history_size(uint value) { compute_cache_.set_unwrap_history_size(value); }

    inline void set_is_computation_stopped(bool value) { compute_cache_.set_is_computation_stopped(value); }

    inline void set_renorm_enabled(bool value) { view_cache_.set_renorm_enabled(value); }

    // RGB
    inline void set_rgb(CompositeRGB value) { composite_cache_.set_rgb(value); }

    void set_rgb_p(Span<int> span, bool notify = false);

    inline void set_weight_r(float value) { composite_cache_.get_rgb_ref()->weight.r = value; }
    inline void set_weight_g(float value) { composite_cache_.get_rgb_ref()->weight.g = value; }
    inline void set_weight_b(float value) { composite_cache_.get_rgb_ref()->weight.b = value; }

    void set_weight_rgb(int r, int g, int b);

    // HSV
    void set_composite_p_h(Span<uint> span, bool notify = false);

    inline void set_hsv(CompositeHSV value) { composite_cache_.set_hsv(value); }
    inline void set_slider_h_threshold_min(float value)
    {
        composite_cache_.get_hsv_ref()->h.slider_threshold.min = value;
    }
    inline void set_slider_h_threshold_max(float value)
    {
        composite_cache_.get_hsv_ref()->h.slider_threshold.max = value;
    }
    inline void set_composite_low_h_threshold(float value) { composite_cache_.get_hsv_ref()->h.threshold.min = value; }
    inline void set_composite_high_h_threshold(float value) { composite_cache_.get_hsv_ref()->h.threshold.max = value; }
    inline void set_composite_p_min_h(uint value) { composite_cache_.get_hsv_ref()->h.p.min = value; }
    inline void set_composite_p_max_h(uint value) { composite_cache_.get_hsv_ref()->h.p.max = value; }
    inline void set_h_blur_kernel_size(uint value) { composite_cache_.get_hsv_ref()->h.blur.kernel_size = value; }
    inline void set_composite_p_min_s(uint value) { composite_cache_.get_hsv_ref()->s.p.min = value; }
    inline void set_composite_p_max_s(uint value) { composite_cache_.get_hsv_ref()->s.p.max = value; }
    inline void set_slider_s_threshold_min(float value)
    {
        composite_cache_.get_hsv_ref()->s.slider_threshold.min = value;
    }
    inline void set_slider_s_threshold_max(float value)
    {
        composite_cache_.get_hsv_ref()->s.slider_threshold.max = value;
    }
    inline void set_composite_low_s_threshold(float value) { composite_cache_.get_hsv_ref()->s.threshold.min = value; }
    inline void set_composite_high_s_threshold(float value) { composite_cache_.get_hsv_ref()->s.threshold.max = value; }
    inline void set_composite_p_min_v(uint value) { composite_cache_.get_hsv_ref()->v.p.min = value; }
    inline void set_composite_p_max_v(uint value) { composite_cache_.get_hsv_ref()->v.p.max = value; }
    inline void set_slider_v_threshold_min(float value)
    {
        composite_cache_.get_hsv_ref()->v.slider_threshold.min = value;
    }
    inline void set_slider_v_threshold_max(float value)
    {
        composite_cache_.get_hsv_ref()->v.slider_threshold.max = value;
    }
    inline void set_composite_low_v_threshold(float value) { composite_cache_.get_hsv_ref()->v.threshold.min = value; }
    inline void set_composite_high_v_threshold(float value) { composite_cache_.get_hsv_ref()->v.threshold.max = value; }
    inline void set_h_blur_activated(bool value) { composite_cache_.get_hsv_ref()->h.blur.enabled = value; }
    inline void set_composite_p_activated_s(bool value) { composite_cache_.get_hsv_ref()->s.p.activated = value; }
    inline void set_composite_p_activated_v(bool value) { composite_cache_.get_hsv_ref()->v.p.activated = value; }

    inline void set_reticle_scale(float value) { view_cache_.set_reticle_scale(value); }

    inline void set_time_transformation_cuts_output_buffer_size(uint value)
    {
        compute_cache_.set_time_transformation_cuts_output_buffer_size(value);
    }

    inline void set_filter2d_smooth_low(int value) { filter2d_cache_.set_filter2d_smooth_low(value); }

    inline void set_filter2d_smooth_high(int value) { filter2d_cache_.set_filter2d_smooth_high(value); }

    inline void set_contrast_lower_threshold(float value) { advanced_cache_.set_contrast_lower_threshold(value); }

    inline void set_contrast_upper_threshold(float value) { advanced_cache_.set_contrast_upper_threshold(value); }

    inline void set_renorm_constant(unsigned value) { advanced_cache_.set_renorm_constant(value); }

    inline void set_cuts_contrast_p_offset(uint value) { advanced_cache_.set_cuts_contrast_p_offset(value); }

    inline void set_reticle_display_enabled(bool value) { view_cache_.set_reticle_display_enabled(value); }

    inline void set_raw_bitshift(unsigned int value) { advanced_cache_.set_raw_bitshift(value); }

    inline void set_signal_zone(units::RectFd value) { zone_cache_.set_signal_zone(value); }
    inline void set_noise_zone(units::RectFd value) { zone_cache_.set_noise_zone(value); }
    inline void set_composite_zone(units::RectFd value) { zone_cache_.set_composite_zone(value); }
    inline void set_zoomed_zone(units::RectFd value) { zone_cache_.set_zoomed_zone(value); }
    inline void set_reticle_zone(units::RectFd value) { zone_cache_.set_reticle_zone(value); }

    enum class ComputeSettingsVersion
    {
        V4,
        InternalV4
    };

    void load_compute_settings(const std::string& settings_path);
    void load_compute_settings(const std::string& settings_path, ComputeSettingsVersion ver, bool no_converter = false);

    void save_compute_settings(const std::string& settings_path);
    void save_compute_settings(const std::string& settings_path, ComputeSettingsVersion ver, bool no_converter = false);

#pragma endregion
    void change_window(uint index);

    const ViewWindow& get_current_window() const;
    const ViewWindow& get_window(WindowKind kind) const;

    void set_notify_callback(std::function<void()> func) { notify_callback_ = func; }

    void update_contrast(WindowKind kind, float min, float max);

    void notify() { notify_callback_(); }

  private:
    GSH() noexcept {}

    std::shared_ptr<holovibes::ViewWindow> get_window(WindowKind kind);
    std::shared_ptr<holovibes::ViewWindow> get_current_window();

    std::function<void()> notify_callback_ = []() {};

    ComputeCache::Ref compute_cache_;
    CompositeCache::Ref composite_cache_;
    ExportCache::Ref export_cache_;
    ImportCache::Ref import_cache_;
    Filter2DCache::Ref filter2d_cache_;
    ViewCache::Ref view_cache_;
    AdvancedCache::Ref advanced_cache_;
    FileReadCache::Ref file_read_cache_;
    ZoneCache::Ref zone_cache_;

    mutable std::mutex mutex_;
};

} // namespace holovibes
