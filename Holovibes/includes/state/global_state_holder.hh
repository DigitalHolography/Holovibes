#pragma once

#include <mutex>

#include "fast_updates_holder.hh"
#include "caches.hh"
#include "entities.hh"

namespace holovibes
{

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
 * MicroCache : local state holder belonging to each worker. Previously, each worker had to fetch data at the same
 * place ; therefore, all variables had to be atomic, with the aim to be thread safe. In order to avoid that, each
 * worker now possess MicroCaches, containing all the state they need. Those MicroCaches are accessed only by their
 * worker, and are synchronized with the GSH when each worker chooses to (using a trigger system).
 * More informations and explanations concerning their synchronization with the GSH are provided in files micro-cache.hh
 * and micro-cache.hxx.
 *
 * FastUpdateHolder : the fastUpdateHolder is a templated map which is used by the informationWorker to access and
 * display information (like fps and queue occupancy) at a high rate, since this needs to be updated continuously.
 */
class GSH
{
  public:
    GSH(GSH& other) = delete;
    void operator=(const GSH&) = delete;

    static GSH& instance();

    // inline prevents MSVC from brain-dying, dunno why
    template <class T>
    static inline FastUpdatesHolder<T> fast_updates_map;

#pragma region(collapsed) GETTERS

    uint get_time_transformation_stride() const noexcept { return compute_cache_.get_time_transformation_stride(); }

    SpaceTransformation get_space_transformation() const noexcept { return compute_cache_.get_space_transformation(); }

    TimeTransformation get_time_transformation() const noexcept { return compute_cache_.get_time_transformation(); };

    uint get_batch_size() const noexcept { return compute_cache_.get_batch_size(); }
    uint get_time_transformation_size() const noexcept { return compute_cache_.get_time_transformation_size(); }
    float get_lambda() const noexcept { return compute_cache_.get_lambda(); }
    float get_z_distance() const noexcept { return compute_cache_.get_z_distance(); };
    bool get_convolution_enabled() const noexcept { return compute_cache_.get_convolution_enabled(); }
    const std::vector<float>& get_convo_matrix_const_ref() { return compute_cache_.get_convo_matrix_const_ref(); };

    int get_filter2d_n1() const noexcept { return filter2d_cache_.get_filter2d_n1(); }
    int get_filter2d_n2() const noexcept { return filter2d_cache_.get_filter2d_n2(); }
    ImgType get_img_type() const noexcept { return view_cache_.get_img_type(); }

    View_XY get_x() const noexcept { return view_cache_.get_x(); }
    int get_x_accu_level() const noexcept { return view_cache_.get_x().accu_level; }
    int get_x_cuts() const noexcept { return view_cache_.get_x().cuts; }

    View_XY get_y() const noexcept { return view_cache_.get_y(); }
    int get_y_accu_level() const noexcept { return view_cache_.get_y().accu_level; }
    int get_y_cuts() const noexcept { return view_cache_.get_y().cuts; }

    View_PQ get_p() const noexcept { return view_cache_.get_p(); }
    int get_p_accu_level() const noexcept { return view_cache_.get_p().accu_level; }
    uint get_p_index() const noexcept { return view_cache_.get_p().index; }

    View_PQ get_q() const noexcept { return view_cache_.get_q(); }
    int get_q_accu_level() const noexcept { return view_cache_.get_q().accu_level; }
    uint get_q_index() const noexcept { return view_cache_.get_q().index; }

    View_XYZ get_xy() const noexcept { return view_cache_.get_xy(); }
    bool get_xy_flip_enabled() const noexcept { return view_cache_.get_xy().flip_enabled; }
    float get_xy_rot() const noexcept { return view_cache_.get_xy().flip_enabled; }
    uint get_xy_img_accu_level() const noexcept { return view_cache_.get_xy().img_accu_level; }
    bool get_xy_log_scale_slice_enabled() const noexcept { return view_cache_.get_xy().log_scale_slice_enabled; }
    bool get_xy_contrast_enabled() const noexcept { return view_cache_.get_xy().contrast_enabled; }
    bool get_xy_contrast_auto_refresh() const noexcept { return view_cache_.get_xy().contrast_auto_refresh; }
    bool get_xy_contrast_invert() const noexcept { return view_cache_.get_xy().contrast_invert; }
    float get_xy_contrast_min() const noexcept { return view_cache_.get_xy().contrast_min; }
    float get_xy_contrast_max() const noexcept { return view_cache_.get_xy().contrast_max; }
    bool get_xy_img_accu_enabled() const noexcept { return view_cache_.get_xy().img_accu_level > 1; }

    View_XYZ get_xz() const noexcept { return view_cache_.get_xz(); }
    bool get_xz_flip_enabled() const noexcept { return view_cache_.get_xz().flip_enabled; }
    float get_xz_rot() const noexcept { return view_cache_.get_xz().flip_enabled; }
    uint get_xz_img_accu_level() const noexcept { return view_cache_.get_xz().img_accu_level; }
    bool get_xz_log_scale_slice_enabled() const noexcept { return view_cache_.get_xz().log_scale_slice_enabled; }
    bool get_xz_contrast_enabled() const noexcept { return view_cache_.get_xz().contrast_enabled; }
    bool get_xz_contrast_auto_refresh() const noexcept { return view_cache_.get_xz().contrast_auto_refresh; }
    bool get_xz_contrast_invert() const noexcept { return view_cache_.get_xz().contrast_invert; }
    float get_xz_contrast_min() const noexcept { return view_cache_.get_xz().contrast_min; }
    float get_xz_contrast_max() const noexcept { return view_cache_.get_xz().contrast_max; }
    bool get_xz_img_accu_enabled() const noexcept { return view_cache_.get_xz().img_accu_level > 1; }

    View_XYZ get_yz() const noexcept { return view_cache_.get_yz(); }
    bool get_yz_flip_enabled() const noexcept { return view_cache_.get_yz().flip_enabled; }
    float get_yz_rot() const noexcept { return view_cache_.get_yz().flip_enabled; }
    uint get_yz_img_accu_level() const noexcept { return view_cache_.get_yz().img_accu_level; }
    bool get_yz_log_scale_slice_enabled() const noexcept { return view_cache_.get_yz().log_scale_slice_enabled; }
    bool get_yz_contrast_enabled() const noexcept { return view_cache_.get_yz().contrast_enabled; }
    bool get_yz_contrast_auto_refresh() const noexcept { return view_cache_.get_yz().contrast_auto_refresh; }
    bool get_yz_contrast_invert() const noexcept { return view_cache_.get_yz().contrast_invert; }
    float get_yz_contrast_min() const noexcept { return view_cache_.get_yz().contrast_min; }
    float get_yz_contrast_max() const noexcept { return view_cache_.get_yz().contrast_max; }
    bool get_yz_img_accu_enabled() const noexcept { return view_cache_.get_yz().img_accu_level > 1; }

    View_Window get_filter2d() const noexcept { return view_cache_.get_filter2d(); }
    bool get_filter2d_contrast_enabled() const noexcept { return view_cache_.get_filter2d().contrast_enabled; }
    bool get_filter2d_contrast_invert() const noexcept { return view_cache_.get_filter2d().contrast_invert; }
    float get_filter2d_contrast_min() const noexcept { return view_cache_.get_filter2d().contrast_min; }
    float get_filter2d_contrast_max() const noexcept { return view_cache_.get_filter2d().contrast_max; }
    bool get_filter2d_log_scale_slice_enabled() const noexcept
    {
        return view_cache_.get_filter2d().log_scale_slice_enabled;
    }
    bool get_filter2d_contrast_auto_refresh() const noexcept
    {
        return view_cache_.get_filter2d().contrast_auto_refresh;
    }

    WindowKind get_current_window_type() const noexcept { return view_cache_.get_current_window(); }

    bool get_contrast_auto_refresh() const noexcept { return get_current_window().contrast_auto_refresh; }
    bool get_contrast_invert() const noexcept { return get_current_window().contrast_invert; }
    bool get_contrast_enabled() const noexcept { return get_current_window().contrast_enabled; }

    bool is_current_window_xyz_type() const;

    const View_Window& get_current_window() const;

    // Over current window
    float get_contrast_min() const;
    float get_contrast_max() const;
    double get_rotation() const;
    bool get_flip_enabled() const;
    bool get_img_log_scale_slice_enabled() const;
    unsigned get_img_accu_level() const;

    bool get_divide_convolution_enabled() const { return compute_cache_.get_divide_convolution_enabled(); };

    bool get_lens_view_enabled() const { return view_cache_.get_lens_view_enabled(); };

    uint get_input_fps() const { return compute_cache_.get_input_fps(); };

    bool get_frame_record_enabled() const { return export_cache_.get_frame_record_enabled(); };

    bool get_chart_display_enabled() const { return view_cache_.get_chart_display_enabled(); };

    bool get_chart_record_enabled() const { return export_cache_.get_chart_record_enabled(); };

    Computation get_compute_mode() const noexcept { return compute_cache_.get_compute_mode(); };

    bool get_filter2d_enabled() const noexcept { return view_cache_.get_filter2d_enabled(); }

    bool get_filter2d_view_enabled() const noexcept { return view_cache_.get_filter2d_view_enabled(); }

    CompositeKind get_composite_kind() const noexcept { return composite_cache_.get_composite_kind(); }

    bool get_fft_shift_enabled() const noexcept { return view_cache_.get_fft_shift_enabled(); }

    bool get_raw_view_enabled() const noexcept { return view_cache_.get_raw_view_enabled(); }

    bool get_composite_auto_weights() const noexcept { return composite_cache_.get_composite_auto_weights(); }

    uint get_start_frame() const noexcept { return import_cache_.get_start_frame(); }
    uint get_end_frame() const noexcept { return import_cache_.get_end_frame(); }

    float get_display_rate() const noexcept { return advanced_cache_.get_display_rate(); }

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
    inline Composite_RGB get_rgb() const noexcept { return composite_cache_.get_rgb(); }
    inline uint get_rgb_p_min() const noexcept { return composite_cache_.get_rgb().p_min; }
    inline uint get_rgb_p_max() const noexcept { return composite_cache_.get_rgb().p_max; }
    inline float get_weight_r() const noexcept { return composite_cache_.get_rgb().weight_r; }
    inline float get_weight_g() const noexcept { return composite_cache_.get_rgb().weight_g; }
    inline float get_weight_b() const noexcept { return composite_cache_.get_rgb().weight_b; }

    // HSV
    inline Composite_HSV get_hsv() const noexcept { return composite_cache_.get_hsv(); }
    inline uint get_composite_p_min_h() const noexcept { return composite_cache_.get_hsv().h.p_min; }
    inline uint get_composite_p_max_h() const noexcept { return composite_cache_.get_hsv().h.p_max; }
    inline float get_slider_h_threshold_min() const noexcept
    {
        return composite_cache_.get_hsv().h.slider_threshold_min;
    }
    inline float get_slider_h_threshold_max() const noexcept
    {
        return composite_cache_.get_hsv().h.slider_threshold_max;
    }
    inline int get_raw_bitshift() const noexcept { return advanced_cache_.get_raw_bitshift(); }

    inline float get_composite_low_h_threshold() const noexcept { return composite_cache_.get_hsv().h.low_threshold; }
    inline float get_composite_high_h_threshold() const noexcept { return composite_cache_.get_hsv().h.high_threshold; }
    inline uint get_h_blur_kernel_size() const noexcept { return composite_cache_.get_hsv().h.blur_kernel_size; }
    inline uint get_composite_p_min_s() const noexcept { return composite_cache_.get_hsv().s.p_min; }
    inline uint get_composite_p_max_s() const noexcept { return composite_cache_.get_hsv().s.p_max; }
    inline float get_slider_s_threshold_min() const noexcept
    {
        return composite_cache_.get_hsv().s.slider_threshold_min;
    }
    inline float get_slider_s_threshold_max() const noexcept
    {
        return composite_cache_.get_hsv().s.slider_threshold_max;
    }
    inline float get_composite_low_s_threshold() const noexcept { return composite_cache_.get_hsv().s.low_threshold; }
    inline float get_composite_high_s_threshold() const noexcept { return composite_cache_.get_hsv().s.high_threshold; }
    inline uint get_composite_p_min_v() const noexcept { return composite_cache_.get_hsv().v.p_min; }
    inline uint get_composite_p_max_v() const noexcept { return composite_cache_.get_hsv().v.p_max; }
    inline float get_slider_v_threshold_min() const noexcept
    {
        return composite_cache_.get_hsv().v.slider_threshold_min;
    }
    inline float get_slider_v_threshold_max() const noexcept
    {
        return composite_cache_.get_hsv().v.slider_threshold_max;
    }
    inline float get_composite_low_v_threshold() const noexcept { return composite_cache_.get_hsv().v.low_threshold; }
    inline float get_composite_high_v_threshold() const noexcept { return composite_cache_.get_hsv().v.high_threshold; }
    inline bool get_h_blur_activated() const noexcept { return composite_cache_.get_hsv().h.blur_enabled; }
    inline bool get_composite_p_activated_s() const noexcept { return composite_cache_.get_hsv().s.p_activated; }
    inline bool get_composite_p_activated_v() const noexcept { return composite_cache_.get_hsv().v.p_activated; }

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
    void set_time_transformation_stride(uint value);
    void disable_convolution();
    void enable_convolution(const std::string& file);
    void set_convolution_enabled(bool value);

    void set_space_transformation(const SpaceTransformation value) noexcept
    {
        compute_cache_.set_space_transformation(value);
    }

    void set_time_transformation(const TimeTransformation value) noexcept
    {
        compute_cache_.set_time_transformation(value);
    }

    void set_lambda(float value) noexcept { compute_cache_.set_lambda(value); }

    void set_z_distance(float value) noexcept { compute_cache_.set_z_distance(value); }

    void set_filter2d_n1(int value) noexcept { filter2d_cache_.set_filter2d_n1(value); }
    void set_filter2d_n2(int value) noexcept { filter2d_cache_.set_filter2d_n2(value); }

    void set_img_type(ImgType value) noexcept { view_cache_.set_img_type(value); }

    void set_x(View_XY value) noexcept { view_cache_.set_x(value); }
    void set_x_accu_level(int value) noexcept { view_cache_.get_x_ref()->accu_level = value; }
    void set_x_cuts(int value) noexcept { view_cache_.get_x_ref()->cuts = value; }

    void set_y(View_XY value) noexcept { view_cache_.set_y(value); }
    void set_y_accu_level(int value) noexcept { view_cache_.get_y_ref()->accu_level = value; }
    void set_y_cuts(int value) noexcept { view_cache_.get_y_ref()->cuts = value; }

    void set_p(View_PQ value) noexcept { view_cache_.set_p(value); }
    void set_p_accu_level(int value) noexcept { view_cache_.get_p_ref()->accu_level = value; }
    void set_p_index(uint value) noexcept { view_cache_.get_p_ref()->index = value; }

    void set_q(View_PQ value) noexcept { view_cache_.set_q(value); }
    void set_q_accu_level(int value) noexcept { view_cache_.get_q_ref()->accu_level = value; }
    void set_q_index(uint value) noexcept { view_cache_.get_q_ref()->index = value; }

    void set_xy(View_XYZ value) noexcept { view_cache_.set_xy(value); }
    void set_xy_flip_enabled(bool value) noexcept { view_cache_.get_xy_ref()->flip_enabled = value; }
    void set_xy_rot(float value) noexcept { view_cache_.get_xy_ref()->rot = value; }
    void set_xy_img_accu_level(uint value) noexcept { view_cache_.get_xy_ref()->img_accu_level = value; }
    void set_xy_log_scale_slice_enabled(bool value) noexcept
    {
        view_cache_.get_xy_ref()->log_scale_slice_enabled = value;
    }
    void set_xy_contrast_enabled(bool value) noexcept { view_cache_.get_xy_ref()->contrast_enabled = value; }
    void set_xy_contrast_auto_refresh(bool value) noexcept { view_cache_.get_xy_ref()->contrast_auto_refresh = value; }
    void set_xy_contrast_invert(bool value) noexcept { view_cache_.get_xy_ref()->contrast_invert = value; }
    void set_xy_contrast_min(float value) noexcept
    {
        view_cache_.get_xy_ref()->contrast_min = value > 1.0f ? value : 1.0f;
    }
    void set_xy_contrast_max(float value) noexcept
    {
        view_cache_.get_xy_ref()->contrast_max = value > 1.0f ? value : 1.0f;
    }

    void set_xz(View_XYZ value) noexcept { view_cache_.set_xz(value); }
    void set_xz_flip_enabled(bool value) noexcept { view_cache_.get_xz_ref()->flip_enabled = value; }
    void set_xz_rot(float value) noexcept { view_cache_.get_xz_ref()->rot = value; }
    void set_xz_img_accu_level(uint value) noexcept { view_cache_.get_xz_ref()->img_accu_level = value; }
    void set_xz_log_scale_slice_enabled(bool value) noexcept
    {
        view_cache_.get_xz_ref()->log_scale_slice_enabled = value;
    }
    void set_xz_contrast_enabled(bool value) noexcept { view_cache_.get_xz_ref()->contrast_enabled = value; }
    void set_xz_contrast_auto_refresh(bool value) noexcept { view_cache_.get_xz_ref()->contrast_auto_refresh = value; }
    void set_xz_contrast_invert(bool value) noexcept { view_cache_.get_xz_ref()->contrast_invert = value; }
    void set_xz_contrast_min(float value) noexcept
    {
        view_cache_.get_xz_ref()->contrast_min = value > 1.0f ? value : 1.0f;
    }
    void set_xz_contrast_max(float value) noexcept
    {
        view_cache_.get_xz_ref()->contrast_max = value > 1.0f ? value : 1.0f;
    }

    void set_yz(View_XYZ value) noexcept { view_cache_.set_yz(value); }
    void set_yz_flip_enabled(bool value) noexcept { view_cache_.get_yz_ref()->flip_enabled = value; }
    void set_yz_rot(float value) noexcept { view_cache_.get_yz_ref()->rot = value; }
    void set_yz_img_accu_level(uint value) noexcept { view_cache_.get_yz_ref()->img_accu_level = value; }
    void set_yz_log_scale_slice_enabled(bool value) noexcept
    {
        view_cache_.get_yz_ref()->log_scale_slice_enabled = value;
    }
    void set_yz_contrast_enabled(bool value) noexcept { view_cache_.get_yz_ref()->contrast_enabled = value; }
    void set_yz_contrast_auto_refresh(bool value) noexcept { view_cache_.get_yz_ref()->contrast_auto_refresh = value; }
    void set_yz_contrast_invert(bool value) noexcept { view_cache_.get_yz_ref()->contrast_invert = value; }
    void set_yz_contrast_min(float value) noexcept
    {
        view_cache_.get_yz_ref()->contrast_min = value > 1.0f ? value : 1.0f;
    }
    void set_yz_contrast_max(float value) noexcept
    {
        view_cache_.get_yz_ref()->contrast_max = value > 1.0f ? value : 1.0f;
    }

    void set_filter2d(View_Window value) noexcept { view_cache_.set_filter2d(value); }
    void set_filter2d_log_scale_slice_enabled(bool value) noexcept
    {
        view_cache_.get_filter2d_ref()->log_scale_slice_enabled = value;
    }
    void set_filter2d_contrast_enabled(bool value) noexcept
    {
        view_cache_.get_filter2d_ref()->contrast_enabled = value;
    }
    void set_filter2d_contrast_auto_refresh(bool value) noexcept
    {
        view_cache_.get_filter2d_ref()->contrast_auto_refresh = value;
    }
    void set_filter2d_contrast_invert(bool value) noexcept { view_cache_.get_filter2d_ref()->contrast_invert = value; }
    void set_filter2d_contrast_min(float value) noexcept
    {
        view_cache_.get_filter2d_ref()->contrast_min = value > 1.0f ? value : 1.0f;
    }
    void set_filter2d_contrast_max(float value) noexcept
    {
        view_cache_.get_filter2d_ref()->contrast_max = value > 1.0f ? value : 1.0f;
    }

    void set_log_scale_filter2d_enabled(bool log_scale_filter2d_enabled) noexcept
    {
        view_cache_.get_filter2d_ref()->log_scale_slice_enabled = log_scale_filter2d_enabled;
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

    void set_divide_convolution_enabled(bool value) { compute_cache_.set_divide_convolution_enabled(value); };

    void set_lens_view_enabled(bool value) { view_cache_.set_lens_view_enabled(value); }

    void set_input_fps(uint value) { compute_cache_.set_input_fps(value); };

    void set_frame_record_enabled(bool value) { export_cache_.set_frame_record_enabled(value); }

    void set_chart_display_enabled(bool value) { view_cache_.set_chart_display_enabled(value); }

    void set_chart_record_enabled(bool value) { export_cache_.set_chart_record_enabled(value); }

    void set_compute_mode(Computation value) { compute_cache_.set_compute_mode(value); }

    void set_filter2d_enabled(bool value) { view_cache_.set_filter2d_enabled(value); }

    void set_filter2d_view_enabled(bool value) { view_cache_.set_filter2d_view_enabled(value); }

    void set_composite_kind(CompositeKind value) { composite_cache_.set_composite_kind(value); }

    void set_fft_shift_enabled(bool value);

    void set_raw_view_enabled(bool value) { view_cache_.set_raw_view_enabled(value); }

    void set_composite_auto_weights(bool value) { composite_cache_.set_composite_auto_weights(value); }

    void set_start_frame(uint value) { import_cache_.set_start_frame(value); }

    void set_end_frame(uint value) { import_cache_.set_end_frame(value); }

    void set_display_rate(float value) { advanced_cache_.set_display_rate(value); }

    void set_cuts_view_enabled(bool value) { view_cache_.set_cuts_view_enabled(value); }

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
    inline void set_rgb(Composite_RGB value) { composite_cache_.set_rgb(value); }

    inline void set_rgb_p_min(int value) { composite_cache_.get_rgb_ref()->p_min = value; }
    inline void set_rgb_p_max(int value) { composite_cache_.get_rgb_ref()->p_max = value; }
    inline void set_weight_r(float value) { composite_cache_.get_rgb_ref()->weight_r = value; }
    inline void set_weight_g(float value) { composite_cache_.get_rgb_ref()->weight_g = value; }
    inline void set_weight_b(float value) { composite_cache_.get_rgb_ref()->weight_b = value; }

    void set_weight_rgb(int r, int g, int b);

    // HSV
    inline void set_hsv(Composite_HSV value) { composite_cache_.set_hsv(value); }
    inline void set_composite_p_min_h(uint value) { composite_cache_.get_hsv_ref()->h.p_min = value; }
    inline void set_composite_p_max_h(uint value) { composite_cache_.get_hsv_ref()->h.p_max = value; }
    inline void set_slider_h_threshold_min(float value)
    {
        composite_cache_.get_hsv_ref()->h.slider_threshold_min = value;
    }
    inline void set_slider_h_threshold_max(float value)
    {
        composite_cache_.get_hsv_ref()->h.slider_threshold_max = value;
    }
    inline void set_composite_low_h_threshold(float value) { composite_cache_.get_hsv_ref()->h.low_threshold = value; }
    inline void set_composite_high_h_threshold(float value)
    {
        composite_cache_.get_hsv_ref()->h.high_threshold = value;
    }
    inline void set_h_blur_kernel_size(uint value) { composite_cache_.get_hsv_ref()->h.blur_kernel_size = value; }
    inline void set_composite_p_min_s(uint value) { composite_cache_.get_hsv_ref()->s.p_min = value; }
    inline void set_composite_p_max_s(uint value) { composite_cache_.get_hsv_ref()->s.p_max = value; }
    inline void set_slider_s_threshold_min(float value)
    {
        composite_cache_.get_hsv_ref()->s.slider_threshold_min = value;
    }
    inline void set_slider_s_threshold_max(float value)
    {
        composite_cache_.get_hsv_ref()->s.slider_threshold_max = value;
    }
    inline void set_composite_low_s_threshold(float value) { composite_cache_.get_hsv_ref()->s.low_threshold = value; }
    inline void set_composite_high_s_threshold(float value)
    {
        composite_cache_.get_hsv_ref()->s.high_threshold = value;
    }
    inline void set_composite_p_min_v(uint value) { composite_cache_.get_hsv_ref()->v.p_min = value; }
    inline void set_composite_p_max_v(uint value) { composite_cache_.get_hsv_ref()->v.p_max = value; }
    inline void set_slider_v_threshold_min(float value)
    {
        composite_cache_.get_hsv_ref()->v.slider_threshold_min = value;
    }
    inline void set_slider_v_threshold_max(float value)
    {
        composite_cache_.get_hsv_ref()->v.slider_threshold_max = value;
    }
    inline void set_composite_low_v_threshold(float value) { composite_cache_.get_hsv_ref()->v.low_threshold = value; }
    inline void set_composite_high_v_threshold(float value)
    {
        composite_cache_.get_hsv_ref()->v.high_threshold = value;
    }
    inline void set_h_blur_activated(bool value) { composite_cache_.get_hsv_ref()->h.blur_enabled = value; }
    inline void set_composite_p_activated_s(bool value) { composite_cache_.get_hsv_ref()->s.p_activated = value; }
    inline void set_composite_p_activated_v(bool value) { composite_cache_.get_hsv_ref()->v.p_activated = value; }

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

    inline void set_raw_bitshift(int value) { advanced_cache_.set_raw_bitshift(value); }

    inline void set_signal_zone(units::RectFd value) { zone_cache_.set_signal_zone(value); }
    inline void set_noise_zone(units::RectFd value) { zone_cache_.set_noise_zone(value); }
    inline void set_composite_zone(units::RectFd value) { zone_cache_.set_composite_zone(value); }
    inline void set_zoomed_zone(units::RectFd value) { zone_cache_.set_zoomed_zone(value); }
    inline void set_reticle_zone(units::RectFd value) { zone_cache_.set_reticle_zone(value); }

#pragma endregion
    void change_window(uint index);

  private:
    GSH() noexcept {}

    std::shared_ptr<holovibes::View_Window> get_current_window();

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
