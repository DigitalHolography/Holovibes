/*! \file
 *
 * \brief Contains compute parameters.
 */
#pragma once

#include <atomic>
#include <mutex>
#include "aliases.hh"
#include "observable.hh"
#include "rect.hh"

// enum
#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_computation.hh"
#include "enum_access_mode.hh"
#include "enum_window_kind.hh"
#include "enum_composite_kind.hh"

// struct
#include "composite_struct.hh"
#include "view_struct.hh"

namespace holovibes
{
/*! \class ComputeDescriptor
 *
 * \brief Contains compute parameters.
 *
 * Theses parameters will be used when the pipe is refresh.
 * It defines parameters for FFT, lens (Fresnel transforms ...),
 * post-processing (contrast, shift_corners, log scale).
 *
 * The class use the *Observer* design pattern instead of the signal
 * mechanism of Qt because classes in the namespace holovibes are
 * independent of GUI or CLI implementations. So that, the code remains
 * reusable.
 *
 * This class contains std::atomic fields to avoid concurrent access between
 * the pipe and the GUI.
 */
class ComputeDescriptor : public Observable
{
  private:
    /*! \brief The lock used in the zone accessors */
    mutable std::mutex mutex_;

    /*! \brief The zone for the signal chart */
    units::RectFd signal_zone;
    /*! \brief The zone for the noise chart */
    units::RectFd noise_zone;
    /*! \brief The area on which we'll normalize the colors */
    units::RectFd composite_zone;
    /*! \brief The area used to limit the stft computations. */
    units::RectFd zoomed_zone;
    /*! \brief The zone of the reticle area */
    units::RectFd reticle_zone;

  public:
    /*! \brief ComputeDescriptor constructor
     * Initialize the compute descriptor to default values of computation. */
    ComputeDescriptor();

    /*! \brief ComputeDescriptor destructor. */
    ~ComputeDescriptor();

    /*! \brief Assignment operator
     * The assignment operator is explicitely defined because std::atomic type
     * does not allow to generate assignments operator automatically.
     */
    ComputeDescriptor& operator=(const ComputeDescriptor& cd);

    /*! \name Accessor to the selected zone
     * \{
     */
    /*!
     * \param	rect The rectangle to process
     * \param m An AccessMode to process.
     */
    void signalZone(units::RectFd& rect, AccessMode m);
    /*!
     * \param	rect The rectangle to process
     * \param m An AccessMode to process.
     */
    void noiseZone(units::RectFd& rect, AccessMode m);
    /*! \} */

    // #############################################
    //  GETTER AND SETTER ZONE
    // #############################################

    /*
    inline double get_rotation() const
    {
        auto w = reinterpret_cast<View_XYZ*>(current);
        return w->rot;
    }
    inline bool get_flip_enabled() const
    {
        auto w = reinterpret_cast<View_XYZ*>(current);
        return w->flip_enabled;
    }
    */

    inline Computation get_compute_mode() const { return compute_mode; }
    inline void set_compute_mode(Computation compute_mode) { this->compute_mode = compute_mode; }

    // inline WindowKind get_current_window() const { return current_window; }

    inline float get_contrast_lower_threshold() const { return contrast_lower_threshold; }
    inline void set_contrast_lower_threshold(float contrast_lower_threshold)
    {
        this->contrast_lower_threshold = contrast_lower_threshold;
    }

    inline float get_contrast_upper_threshold() const { return contrast_upper_threshold; }
    inline void set_contrast_upper_threshold(float contrast_upper_threshold)
    {
        this->contrast_upper_threshold = contrast_upper_threshold;
    }

    inline uint get_cuts_contrast_p_offset() const { return cuts_contrast_p_offset; }

    inline void set_cuts_contrast_p_offset(uint cuts_contrast_p_offset)
    {
        this->cuts_contrast_p_offset = cuts_contrast_p_offset;
    }

    inline float get_pixel_size() const { return pixel_size; }
    inline void set_pixel_size(float pixel_size) { this->pixel_size = pixel_size; }

    inline unsigned get_renorm_constant() const { return renorm_constant; }
    inline void set_renorm_constant(unsigned renorm_constant) { this->renorm_constant = renorm_constant; }

    inline int get_filter2d_smooth_low() const { return filter2d_smooth_low; }
    inline void set_filter2d_smooth_low(int filter2d_smooth_low) { this->filter2d_smooth_low = filter2d_smooth_low; }

    inline int get_filter2d_smooth_high() const { return filter2d_smooth_high; }
    inline void set_filter2d_smooth_high(int filter2d_smooth_high)
    {
        this->filter2d_smooth_high = filter2d_smooth_high;
    }

    inline uint get_file_buffer_size() const { return file_buffer_size; }
    inline void set_file_buffer_size(uint file_buffer_size) { this->file_buffer_size = file_buffer_size; }

    inline uint get_input_buffer_size() const { return input_buffer_size; }
    inline void set_input_buffer_size(uint input_buffer_size) { this->input_buffer_size = input_buffer_size; }

    inline uint get_record_buffer_size() const { return record_buffer_size; }
    inline void set_record_buffer_size(uint record_buffer_size) { this->record_buffer_size = record_buffer_size; }

    inline uint get_output_buffer_size() const { return output_buffer_size; }
    inline void set_output_buffer_size(uint value) { output_buffer_size = value; }

    inline uint get_time_transformation_cuts_output_buffer_size() const
    {
        return time_transformation_cuts_output_buffer_size;
    }
    inline void set_time_transformation_cuts_output_buffer_size(uint value)
    {
        time_transformation_cuts_output_buffer_size = value;
    }

    inline float get_display_rate() const { return display_rate; }
    inline void set_display_rate(float display_rate) { this->display_rate = display_rate; }
    /*
        inline uint get_img_accu_xy_level() const { return xy.img_accu_level.load(); }
        inline void set_img_accu_xy_level(uint img_accu_slice_xy_level)
        {
            this->xy.img_accu_level = img_accu_slice_xy_level;
        }

        inline uint get_img_accu_xz_level() const { return xz.img_accu_level.load(); }
        inline void set_img_accu_xz_level(uint img_accu_slice_xz_level)
        {
            this->xz.img_accu_level = img_accu_slice_xz_level;
        }

        inline uint get_img_accu_yz_level() const { return yz.img_accu_level.load(); }
        inline void set_img_accu_yz_level(uint img_accu_slice_yz_level)
        {
            this->yz.img_accu_level = img_accu_slice_yz_level;
        }
    */
    inline float get_reticle_scale() const { return reticle_scale; }
    inline void set_reticle_scale(float reticle_scale) { this->reticle_scale = reticle_scale; }

    inline uint get_raw_bitshift() const { return raw_bitshift; }
    inline void set_raw_bitshift(uint raw_bitshift) { this->raw_bitshift = raw_bitshift; }

    inline CompositeKind get_composite_kind() const { return composite_kind; }
    inline void set_composite_kind(CompositeKind composite_kind) { this->composite_kind = composite_kind; }

    // RGB
    inline uint get_rgb_p_min() const { return rgb.p_min; }
    inline void set_rgb_p_min(uint composite_p_red) { this->rgb.p_min = composite_p_red; }

    inline uint get_rgb_p_max() const { return rgb.p_max; }
    inline void set_rgb_p_max(uint composite_p_blue) { this->rgb.p_max = composite_p_blue; }

    inline float get_weight_r() const { return rgb.weight_r; }
    inline void set_weight_r(float weight_r) { this->rgb.weight_r = weight_r; }

    inline float get_weight_g() const { return rgb.weight_g; }
    inline void set_weight_g(float weight_g) { this->rgb.weight_g = weight_g; }

    inline float get_weight_b() const { return rgb.weight_b; }
    inline void set_weight_b(float weight_b) { this->rgb.weight_b = weight_b; }

    // HSV
    inline uint get_composite_p_min_h() const { return hsv.h.p_min; }
    inline void set_composite_p_min_h(uint composite_p_min_h) { this->hsv.h.p_min = composite_p_min_h; }

    inline uint get_composite_p_max_h() const { return hsv.h.p_max; }
    inline void set_composite_p_max_h(uint composite_p_max_h) { this->hsv.h.p_max = composite_p_max_h; }

    inline float get_slider_h_threshold_min() const { return hsv.h.slider_threshold_min; }
    inline void set_slider_h_threshold_min(float slider_h_threshold_min)
    {
        this->hsv.h.slider_threshold_min = slider_h_threshold_min;
    }

    inline float get_slider_h_threshold_max() const { return hsv.h.slider_threshold_max; }
    inline void set_slider_h_threshold_max(float slider_h_threshold_max)
    {
        this->hsv.h.slider_threshold_max = slider_h_threshold_max;
    }

    inline float get_composite_low_h_threshold() const { return hsv.h.low_threshold; }
    inline void set_composite_low_h_threshold(float composite_low_h_threshold)
    {
        this->hsv.h.low_threshold = composite_low_h_threshold;
    }

    inline float get_composite_high_h_threshold() const { return hsv.h.high_threshold; }
    inline void set_composite_high_h_threshold(float composite_high_h_threshold)
    {
        this->hsv.h.high_threshold = composite_high_h_threshold;
    }

    inline uint get_h_blur_kernel_size() const { return hsv.h.blur_kernel_size; }
    inline void set_h_blur_kernel_size(uint h_blur_kernel_size) { this->hsv.h.blur_kernel_size = h_blur_kernel_size; }

    inline uint get_composite_p_min_s() const { return hsv.s.p_min; }
    inline void set_composite_p_min_s(uint composite_p_min_s) { this->hsv.s.p_min = composite_p_min_s; }

    inline uint get_composite_p_max_s() const { return hsv.s.p_max; }
    inline void set_composite_p_max_s(uint composite_p_max_s) { this->hsv.s.p_max = composite_p_max_s; }

    inline float get_slider_s_threshold_min() const { return hsv.s.slider_threshold_min; }
    inline void set_slider_s_threshold_min(float slider_s_threshold_min)
    {
        this->hsv.s.slider_threshold_min = slider_s_threshold_min;
    }

    inline float get_slider_s_threshold_max() const { return hsv.s.slider_threshold_max; }
    inline void set_slider_s_threshold_max(float slider_s_threshold_max)
    {
        this->hsv.s.slider_threshold_max = slider_s_threshold_max;
    }

    inline float get_composite_low_s_threshold() const { return hsv.s.low_threshold; }
    inline void set_composite_low_s_threshold(float composite_low_s_threshold)
    {
        this->hsv.s.low_threshold = composite_low_s_threshold;
    }

    inline float get_composite_high_s_threshold() const { return hsv.s.high_threshold; }
    inline void set_composite_high_s_threshold(float composite_high_s_threshold)
    {
        this->hsv.s.high_threshold = composite_high_s_threshold;
    }

    inline uint get_composite_p_min_v() const { return hsv.v.p_min; }
    inline void set_composite_p_min_v(uint composite_p_min_v) { this->hsv.v.p_min = composite_p_min_v; }

    inline uint get_composite_p_max_v() const { return hsv.v.p_max; }
    inline void set_composite_p_max_v(uint composite_p_max_v) { this->hsv.v.p_max = composite_p_max_v; }

    inline float get_slider_v_threshold_min() const { return hsv.v.slider_threshold_min; }
    inline void set_slider_v_threshold_min(float slider_v_threshold_min)
    {
        this->hsv.v.slider_threshold_min = slider_v_threshold_min;
    }

    inline float get_slider_v_threshold_max() const { return hsv.v.slider_threshold_max; }
    inline void set_slider_v_threshold_max(float slider_v_threshold_max)
    {
        this->hsv.v.slider_threshold_max = slider_v_threshold_max;
    }

    inline float get_composite_low_v_threshold() const { return hsv.v.low_threshold; }
    inline void set_composite_low_v_threshold(float composite_low_v_threshold)
    {
        this->hsv.v.low_threshold = composite_low_v_threshold;
    }

    inline float get_composite_high_v_threshold() const { return hsv.v.high_threshold; }
    inline void set_composite_high_v_threshold(float composite_high_v_threshold)
    {
        this->hsv.v.high_threshold = composite_high_v_threshold;
    }

    inline int get_unwrap_history_size() const { return unwrap_history_size; }
    inline void set_unwrap_history_size(int unwrap_history_size) { this->unwrap_history_size = unwrap_history_size; }

    inline bool get_is_computation_stopped() const { return is_computation_stopped; }
    inline void set_is_computation_stopped(bool is_computation_stopped)
    {
        this->is_computation_stopped = is_computation_stopped;
    }

    inline bool get_divide_convolution_enabled() const { return divide_convolution_enabled; }
    inline void set_divide_convolution_enabled(bool divide_convolution_enabled)
    {
        this->divide_convolution_enabled = divide_convolution_enabled;
    }

    inline bool get_renorm_enabled() const { return renorm_enabled; }
    inline void set_renorm_enabled(bool renorm_enabled) { this->renorm_enabled = renorm_enabled; }

    inline bool get_fft_shift_enabled() const { return fft_shift_enabled; }
    inline void set_fft_shift_enabled(bool fft_shift_enabled) { this->fft_shift_enabled = fft_shift_enabled; }

    /*
    inline bool get_contrast_enabled() const { return current->contrast_enabled; }
    inline void set_contrast_enabled(bool contrast_enabled) { current->contrast_enabled = contrast_enabled; }

    inline bool get_contrast_auto_refresh() const { return current->contrast_auto_refresh; }
    inline void set_contrast_auto_refresh(bool contrast_auto_refresh)
    {
        current->contrast_auto_refresh = contrast_auto_refresh;
    }

    inline bool get_contrast_invert() const { return current->contrast_invert; }
    inline void set_contrast_invert(bool contrast_invert) { current->contrast_invert = contrast_invert; }
    */

    inline bool get_filter2d_enabled() const { return filter2d_enabled; }
    inline void set_filter2d_enabled(bool filter2d_enabled) { this->filter2d_enabled = filter2d_enabled; }

    inline bool get_filter2d_view_enabled() const { return filter2d_view_enabled; }
    inline void set_filter2d_view_enabled(bool filter2d_view_enabled)
    {
        this->filter2d_view_enabled = filter2d_view_enabled;
    }

    inline bool get_3d_cuts_view_enabled() const { return time_transformation_cuts_enabled; }
    inline void set_3d_cuts_view_enabled(bool time_transformation_cuts_enabled)
    {
        this->time_transformation_cuts_enabled = time_transformation_cuts_enabled;
    }

    inline bool get_chart_display_enabled() const { return chart_display_enabled; }
    inline void set_chart_display_enabled(bool chart_display_enabled)
    {
        this->chart_display_enabled = chart_display_enabled;
    }

    inline bool get_chart_record_enabled() const { return chart_record_enabled; }
    inline void set_chart_record_enabled(bool chart_record_enabled)
    {
        this->chart_record_enabled = chart_record_enabled;
    }
    /*
        inline bool get_img_accu_xy_enabled() const { return xy.img_accu_level.load() > 1; }
        inline bool get_img_accu_xz_enabled() const { return xz.img_accu_level.load() > 1; }
        inline bool get_img_accu_yz_enabled() const { return yz.img_accu_level.load() > 1; }
    */
    inline bool get_raw_view_enabled() const { return raw_view_enabled; }
    inline void set_raw_view_enabled(bool raw_view_enabled) { this->raw_view_enabled = raw_view_enabled; }

    inline bool get_synchronized_record() const { return synchronized_record; }
    inline void set_synchronized_record(bool synchronized_record) { this->synchronized_record = synchronized_record; }

    inline bool get_reticle_display_enabled() const { return reticle_display_enabled; }
    inline void set_reticle_display_enabled(bool reticle_enabled) { this->reticle_display_enabled = reticle_enabled; }

    inline bool get_h_blur_activated() const { return hsv.h.blur_enabled; }
    inline void set_h_blur_activated(bool h_blur_activated) { this->hsv.h.blur_enabled = h_blur_activated; }

    inline bool get_composite_p_activated_s() const { return hsv.s.p_activated; }
    inline void set_composite_p_activated_s(bool composite_p_activated_s)
    {
        this->hsv.s.p_activated = composite_p_activated_s;
    }

    inline bool get_composite_p_activated_v() const { return hsv.v.p_activated; }
    inline void set_composite_p_activated_v(bool composite_p_activated_v)
    {
        this->hsv.v.p_activated = composite_p_activated_v;
    }

    inline bool get_composite_auto_weights() const { return composite_auto_weights; }
    inline void set_composite_auto_weights(bool composite_auto_weights)
    {
        this->composite_auto_weights = composite_auto_weights;
    }

    inline uint get_start_frame() const { return start_frame; }
    inline void set_start_frame(uint start_frame) { this->start_frame = start_frame; }

    inline uint get_end_frame() const { return end_frame; }
    inline void set_end_frame(uint end_frame) { this->end_frame = end_frame; }

    inline bool get_lens_view_enabled() const { return lens_view_enabled; }
    inline void set_lens_view_enabled(bool value) { lens_view_enabled = value; }

    // #############################################
    //  END GETTER AND SETTER ZONE
    // #############################################

    /*! \name Getter of the overlay positions
     * \{
     */
    units::RectFd getCompositeZone() const;
    units::RectFd getZoomedZone() const;
    units::RectFd getReticleZone() const;
    /*! \} */

    /*! \name	Setter of the overlay positions.
     * \{
     */
    void setCompositeZone(const units::RectFd& rect);
    void setZoomedZone(const units::RectFd& rect);
    void setReticleZone(const units::RectFd& rect);
    /*! \} */

    /*! \brief Get the rounded value of max contrast for the given WindowKind
     *
     * Qt rounds the value by default.
     * In order to compare the compute descriptor values these values also needs to be rounded.
     */
    float get_truncate_contrast_max(const int precision = 2) const;
    /*! \brief Get the rounded value of min contrast for the given WindowKind
     *
     * \see get_truncate_contrast_max()
     */
    float get_truncate_contrast_min(const int precision = 2) const;

    /*! \brief Limit the value of p_index and p_acc according to time_transformation_size */
    void check_p_limits();
    /*! \brief Limit the value of q_index and q_acc according to time_transformation_size */
    void check_q_limits();

    /*! \brief Reset some values after MainWindow receives an update exception */
    void handle_update_exception();
    /*! \brief Reset some values after MainWindow receives an accumulation exception */
    void handle_accumulation_exception();

    void change_angle();
    void change_flip();

    void set_computation_stopped(bool value);

    void set_weight_rgb(int r, int g, int b);

    /*! \brief Reset values used to check if GUY windows are displayed */
    void reset_windows_display();
    /*! \brief Reset values used in the slice view */
    void reset_slice_view();
    /*! \} */

    /*! \name Convolution related operations
     * \{
     */
    void set_convolution(bool enable, const std::string& file);
    void set_divide_by_convo(bool enable);
    void load_convolution_matrix(const std::string& file);
    /*! \brief Input matrix used for convolution */
    std::vector<float> convo_matrix;
    /*! \} */

#pragma region Atomics vars
    // Variables are regroup by module. Those are the same as in the compute_settings.ini
    // Image rendering
    /*! \brief Mode of computation of the image */
    std::atomic<Computation> compute_mode{Computation::Raw};
    /*! \brief Enables filter 2D */
    std::atomic<bool> filter2d_enabled{false};
    /*! \brief Enables filter 2D View */
    std::atomic<bool> filter2d_view_enabled{false};
    /*! \brief Convolution type (file present in AppData) */
    // std::atomic<std::string> convolution_type{""};
    /*! \brief Is divide by convolution enabled */
    std::atomic<bool> divide_convolution_enabled{false};

    // View
    // TODO: Add unwrap2d
    /*! \brief Are slices YZ and XZ enabled */
    std::atomic<bool> time_transformation_cuts_enabled{false};
    /*! \brief Is shift fft enabled (switching representation diagram) */
    std::atomic<bool> fft_shift_enabled{false};
    /*! \brief Is gpu lens display activated */
    std::atomic<bool> lens_view_enabled{false};
    /*! \brief Display the raw interferogram when we are in hologram mode. */
    std::atomic<bool> raw_view_enabled{false};

    /*! \brief Postprocessing renorm enabled */
    std::atomic<bool> renorm_enabled{true};
    /*! \brief Is the reticle overlay enabled */
    std::atomic<bool> reticle_display_enabled{false};
    /*! \brief Reticle border scale */
    std::atomic<float> reticle_scale{0.5f};

    /*! \brief Last window selected */
    std::atomic<WindowKind> current_window{WindowKind::XYview};

    // View_Window* current = &xy;

    // Composite images
    std::atomic<CompositeKind> composite_kind;
    std::atomic<bool> composite_auto_weights;
    Composite_RGB rgb{};
    Composite_HSV hsv{};

    // Advanced
    /*! \brief Max number of frames read each time by the thread_reader. */
    std::atomic<uint> file_buffer_size{512};
    /*! \brief Max size of input queue in number of images. */
    std::atomic<uint> input_buffer_size{512};
    /*! \brief Max size of frame record queue in number of images. */
    std::atomic<uint> record_buffer_size{1024};
    /*! \brief Max size of output queue in number of images. */
    std::atomic<uint> output_buffer_size{256};
    /*! \brief Max size of time transformation cuts queue in number of images. */
    std::atomic<uint> time_transformation_cuts_output_buffer_size{512};
    /*! \brief Number of frame per seconds displayed */
    std::atomic<float> display_rate{30};
    /*! \brief Filter2D low smoothing */ // May be moved in filter2d Struct
    std::atomic<int> filter2d_smooth_low{0};
    /*! \brief Filter2D high smoothing */
    std::atomic<int> filter2d_smooth_high{0};
    std::atomic<float> contrast_lower_threshold{0.5f};
    std::atomic<float> contrast_upper_threshold{99.5f};
    /*! \brief postprocessing remormalize multiplication constant */
    std::atomic<unsigned> renorm_constant{5};
    std::atomic<uint> cuts_contrast_p_offset{2};

    // Other
    /*! \brief Is the computation stopped */
    std::atomic<bool> is_computation_stopped{true};
    /*! \brief Is holovibes currently recording */
    std::atomic<bool> frame_record_enabled{false};
    /*! \brief Wait the beginning of the file to start the recording. */
    std::atomic<bool> synchronized_record{false};
    /*! \brief Max size of unwrapping corrections in number of images.
     *
     * Determines how far, meaning how many iterations back, phase corrections
     * are taken in order to be applied to the current phase image.
     */
    std::atomic<uint> unwrap_history_size{1};

    /*! \brief Size of a pixel in micron */ // Depends on camera or input file.
    std::atomic<float> pixel_size{12.0f};
    /*! \brief Number of bits to shift when in raw mode */
    std::atomic<uint> raw_bitshift{0}; // Never change and surely not used

    /*! \brief First frame read */
    std::atomic<uint> start_frame{0};
    /*! \brief Lasrt frame read */
    std::atomic<uint> end_frame{0};
    /*! \brief The input FPS */
    std::atomic<uint> input_fps{60};

    // Chart
    /*! \brief Enables the signal and noise chart display */
    std::atomic<bool> chart_display_enabled{false};
    /*! \brief Enables the signal and noise chart record */
    std::atomic<bool> chart_record_enabled{false};

#pragma endregion
};
} // namespace holovibes
