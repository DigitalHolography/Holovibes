/*! \file
 *
 * \brief Global State Holder Class
 *
 *  Holds the state of the entire application
 */

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

    bool get_contrast_auto_refresh() const noexcept;
    bool get_contrast_invert() const noexcept;
    bool get_contrast_enabled() const noexcept;

    bool is_current_window_xyz_type() const;

    // Over current window
    float get_contrast_min() const;
    float get_contrast_max() const;
    double get_rotation() const;
    bool get_horizontal_flip() const;
    bool get_log_enabled() const;
    unsigned get_accumulation_level() const;

    inline bool get_divide_convolution_enabled() const { return compute_cache_.get_divide_convolution_enabled(); };

    inline bool get_frame_record_enabled() const { return export_cache_.get_frame_record_enabled(); };

    inline bool get_chart_record_enabled() const { return export_cache_.get_chart_record_enabled(); };

    inline Computation get_compute_mode() const noexcept { return compute_cache_.get_compute_mode(); };

    inline CompositeKind get_composite_kind() const noexcept { return composite_cache_.get_composite_kind(); }

    inline bool get_composite_auto_weights() const noexcept { return composite_cache_.get_composite_auto_weights(); }

    inline uint get_start_frame() const noexcept { return import_cache_.get_start_frame(); }
    inline uint get_end_frame() const noexcept { return import_cache_.get_end_frame(); }

    inline float get_display_rate() const noexcept { return advanced_cache_.get_display_rate(); }

    inline uint get_input_buffer_size() const noexcept { return advanced_cache_.get_input_buffer_size(); }

    inline uint get_record_buffer_size() const noexcept { return advanced_cache_.get_record_buffer_size(); }

    inline uint get_output_buffer_size() const noexcept { return advanced_cache_.get_output_buffer_size(); }

    inline float get_pixel_size() const noexcept { return compute_cache_.get_pixel_size(); }

    inline uint get_unwrap_history_size() const noexcept { return compute_cache_.get_unwrap_history_size(); }

    inline bool get_is_computation_stopped() const noexcept { return compute_cache_.get_is_computation_stopped(); }

    // RGB
    inline CompositeRGB get_rgb() const noexcept { return composite_cache_.get_rgb(); }
    inline uint get_rgb_p_min() const noexcept { return composite_cache_.get_rgb().frame_index.min; }
    inline uint get_rgb_p_max() const noexcept { return composite_cache_.get_rgb().frame_index.max; }
    inline float get_weight_r() const noexcept { return composite_cache_.get_rgb().weight.r; }
    inline float get_weight_g() const noexcept { return composite_cache_.get_rgb().weight.g; }
    inline float get_weight_b() const noexcept { return composite_cache_.get_rgb().weight.b; }

    // HSV
    inline CompositeHSV get_hsv() const noexcept { return composite_cache_.get_hsv(); }
    inline uint get_composite_p_min_h() const noexcept { return composite_cache_.get_hsv().h.frame_index.min; }
    inline uint get_composite_p_max_h() const noexcept { return composite_cache_.get_hsv().h.frame_index.max; }

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
    inline uint get_composite_p_min_s() const noexcept { return composite_cache_.get_hsv().s.frame_index.min; }
    inline uint get_composite_p_max_s() const noexcept { return composite_cache_.get_hsv().s.frame_index.max; }
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
    inline uint get_composite_p_min_v() const noexcept { return composite_cache_.get_hsv().v.frame_index.min; }
    inline uint get_composite_p_max_v() const noexcept { return composite_cache_.get_hsv().v.frame_index.max; }
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
    inline bool get_composite_p_activated_s() const noexcept
    {
        return composite_cache_.get_hsv().s.frame_index.activated;
    }
    inline bool get_composite_p_activated_v() const noexcept
    {
        return composite_cache_.get_hsv().v.frame_index.activated;
    }

    inline uint get_time_transformation_cuts_output_buffer_size() const noexcept
    {
        return compute_cache_.get_time_transformation_cuts_output_buffer_size();
    }

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

    // Over current window
    void set_contrast_enabled(bool contrast_enabled);
    void set_contrast_auto_refresh(bool contrast_auto_refresh);
    void set_contrast_invert(bool contrast_invert);
    void set_contrast_min(float value);
    void set_contrast_max(float value);
    void set_log_enabled(bool value);
    void set_accumulation_level(int value);
    void set_rotation(double value);
    void set_horizontal_flip(double value);

    inline void set_divide_convolution_enabled(bool value) { compute_cache_.set_divide_convolution_enabled(value); };

    inline void set_frame_record_enabled(bool value) { export_cache_.set_frame_record_enabled(value); }

    inline void set_chart_record_enabled(bool value) { export_cache_.set_chart_record_enabled(value); }

    inline void set_compute_mode(Computation value) { compute_cache_.set_compute_mode(value); }

    inline void set_composite_kind(CompositeKind value) { composite_cache_.set_composite_kind(value); }

    inline void set_composite_auto_weights(bool value) { composite_cache_.set_composite_auto_weights(value); }

    inline void set_start_frame(uint value) { import_cache_.set_start_frame(value); }

    inline void set_end_frame(uint value) { import_cache_.set_end_frame(value); }

    inline void set_display_rate(float value) { advanced_cache_.set_display_rate(value); }

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

    // RGB
    inline void set_rgb(CompositeRGB value) { composite_cache_.set_rgb(value); }

    void set_rgb_p(Span<int> span, bool notify = false);

    inline void set_weight_r(double value) { composite_cache_.get_rgb_ref()->weight.r = value; }
    inline void set_weight_g(double value) { composite_cache_.get_rgb_ref()->weight.g = value; }
    inline void set_weight_b(double value) { composite_cache_.get_rgb_ref()->weight.b = value; }

    void set_weight_rgb(double r, double g, double b);

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
    inline void set_composite_p_min_h(uint value) { composite_cache_.get_hsv_ref()->h.frame_index.min = value; }
    inline void set_composite_p_max_h(uint value) { composite_cache_.get_hsv_ref()->h.frame_index.max = value; }
    inline void set_h_blur_kernel_size(uint value) { composite_cache_.get_hsv_ref()->h.blur.kernel_size = value; }
    inline void set_composite_p_min_s(uint value) { composite_cache_.get_hsv_ref()->s.frame_index.min = value; }
    inline void set_composite_p_max_s(uint value) { composite_cache_.get_hsv_ref()->s.frame_index.max = value; }
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
    inline void set_composite_p_min_v(uint value) { composite_cache_.get_hsv_ref()->v.frame_index.min = value; }
    inline void set_composite_p_max_v(uint value) { composite_cache_.get_hsv_ref()->v.frame_index.max = value; }
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
    inline void set_composite_p_activated_s(bool value)
    {
        composite_cache_.get_hsv_ref()->s.frame_index.activated = value;
    }
    inline void set_composite_p_activated_v(bool value)
    {
        composite_cache_.get_hsv_ref()->v.frame_index.activated = value;
    }

    inline void set_time_transformation_cuts_output_buffer_size(uint value)
    {
        compute_cache_.set_time_transformation_cuts_output_buffer_size(value);
    }

    inline void set_contrast_lower_threshold(float value) { advanced_cache_.set_contrast_lower_threshold(value); }

    inline void set_contrast_upper_threshold(float value) { advanced_cache_.set_contrast_upper_threshold(value); }

    inline void set_renorm_constant(unsigned value) { advanced_cache_.set_renorm_constant(value); }

    inline void set_cuts_contrast_p_offset(uint value) { advanced_cache_.set_cuts_contrast_p_offset(value); }

    inline void set_raw_bitshift(unsigned int value) { advanced_cache_.set_raw_bitshift(value); }

    inline void set_signal_zone(units::RectFd value) { zone_cache_.set_signal_zone(value); }
    inline void set_noise_zone(units::RectFd value) { zone_cache_.set_noise_zone(value); }
    inline void set_composite_zone(units::RectFd value) { zone_cache_.set_composite_zone(value); }
    inline void set_zoomed_zone(units::RectFd value) { zone_cache_.set_zoomed_zone(value); }
    inline void set_reticle_zone(units::RectFd value) { zone_cache_.set_reticle_zone(value); }

    enum class ComputeSettingsVersion
    {
        V2,
        V3,
        V4,
        V5
    };
    static void convert_json(json& data, GSH::ComputeSettingsVersion from);

#pragma endregion

    void set_notify_callback(std::function<void()> func) { notify_callback_ = func; }

    void update_contrast(WindowKind kind, float min, float max);

  private:
    GSH() noexcept {}

    std::shared_ptr<holovibes::ViewWindow> get_window(WindowKind kind);

    std::function<void()> notify_callback_ = []() {};
    void notify() { notify_callback_(); }

    ComputeCache::Ref compute_cache_;
    CompositeCache::Ref composite_cache_;
    ExportCache::Ref export_cache_;
    ImportCache::Ref import_cache_;
    AdvancedCache::Ref advanced_cache_;
    FileReadCache::Ref file_read_cache_;
    ZoneCache::Ref zone_cache_;

    mutable std::mutex mutex_;
};

} // namespace holovibes
