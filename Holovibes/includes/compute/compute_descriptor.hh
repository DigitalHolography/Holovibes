/*! \file
 *
 * \brief Contains compute parameters.
 */
#pragma once

#include <atomic>
#include <mutex>
#include "observable.hh"
#include "rect.hh"

// enum
#include "enum_space_transformation.hh"
#include "enum_time_transformation.hh"
#include "enum_computation.hh"
#include "enum_img_type.hh"
#include "enum_access_mode.hh"
#include "enum_window_kind.hh"
#include "enum_composite_kind.hh"

// struct
#include "composite_struct.hh"
#include "view_struct.hh"

namespace holovibes
{

#define __HOLOVIBES_VERSION__ "10.2"
#define __APPNAME__ "Holovibes"

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
    typedef unsigned char uchar;
    typedef unsigned short ushort;
    typedef unsigned int uint;
    typedef unsigned long ulong;

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

    /*! \name General getters / setters to avoid code duplication
     * \{
     */
    float get_contrast_min() const;
    float get_contrast_max() const;

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

    bool get_img_log_scale_slice_enabled() const;

    bool get_img_acc_slice_enabled() const;
    unsigned get_img_acc_slice_level() const;

    bool get_contrast_enabled() const;
    bool get_contrast_auto_refresh() const;
    bool get_contrast_invert_enabled() const;

    void set_contrast_min(float value);
    void set_contrast_max(float value);
    void set_log_scale_slice_enabled(bool value);
    void set_log_scale_slice_enabled_filter2d() { filter2d.log_scale_slice_enabled = true; }
    void set_accumulation(bool value);
    void set_accumulation_level(float value);

    /*! \brief Limit the value of p_index and p_acc according to time_transformation_size */
    void check_p_limits();
    /*! \brief Limit the value of q_index and q_acc according to time_transformation_size */
    void check_q_limits();
    /*! \brief Limit the value of batch_size according to input_queue_capacity */
    void check_batch_size_limit(const uint input_queue_capacity);
    /*! \brief Limit the value of time_transformation_stride according to batch_size or adapt into a multiple of it */
    void adapt_time_transformation_stride();

    /*! \brief Reset some values after MainWindow receives an update exception */
    void handle_update_exception();
    /*! \brief Reset some values after MainWindow receives an accumulation exception */
    void handle_accumulation_exception();

    void set_compute_mode(Computation mode);
    void set_space_transformation_from_string(const std::string& value);
    void set_time_transformation_from_string(const std::string& value);
    void set_time_transformation_stride(int value);
    void set_time_transformation_size(int value);
    void set_batch_size(int value);
    void set_contrast_mode(bool value);
    void change_angle();
    void change_flip();
    void set_contrast_invert(bool value);
    void set_contrast_auto_refresh(bool value);
    void set_contrast_enabled(bool value);
    void set_convolution_enabled(bool value);
    void set_divide_convolution_mode(bool value);
    void set_reticle_view_enabled(bool value);
    void set_reticle_scale(double value);
    void set_img_type(ImgType type);
    void set_computation_stopped(bool value);
    void set_time_transformation_cuts_enabled(bool value);
    void set_renorm_enabled(bool value);
    void set_filter2d_enabled(bool value);
    void set_filter2d_n1(int n);
    void set_filter2d_n2(int n);
    void set_fft_shift_enabled(bool value);
    void set_lens_view_enabled(bool value);
    void set_x_cuts(int value);
    void set_y_cuts(int value);
    void set_p_index(int value);
    void set_q_index(int value);
    void set_lambda(float value);
    void set_zdistance(float value);

    void set_rgb_p_min(int value);
    void set_rgb_p_max(int value);
    void set_composite_p_min_h(int value);
    void set_composite_p_max_h(int value);
    void set_composite_p_min_s(int value);
    void set_composite_p_max_s(int value);
    void set_composite_p_min_v(int value);
    void set_composite_p_max_v(int value);
    void set_weight_rgb(int r, int g, int b);
    void set_composite_auto_weights(bool value);
    void set_composite_kind(CompositeKind kind);
    void set_composite_p_activated_s(bool value);
    void set_composite_p_activated_v(bool value);
    void set_h_blur_activated(bool value);
    void set_h_blur_kernel_size(int value);

    void set_p_accu(bool enabled, int level);
    void set_x_accu(bool enabled, int level);
    void set_y_accu(bool enabled, int level);
    void set_q_accu(bool enabled, int level);

    /*! \brief Change the window according to the given index */
    void change_window(int index);
    /*! \brief Set the image rendering ui params */
    void set_rendering_params(float value);
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
    /*! \brief Number of images dequeued from input to gpu_input_queue */
    std::atomic<uint> batch_size{1};
    /*! \brief Number of pipe iterations between two time transformations (STFT/PCA) */
    std::atomic<uint> time_transformation_stride{1};
    /*! \brief Enables filter 2D */
    std::atomic<bool> filter2d_enabled{false};
    /*! \brief Enables filter 2D View */
    std::atomic<bool> filter2d_view_enabled{false};
    /*! \brief Filter2D low radius */
    std::atomic<int> filter2d_n1{0};
    /*! \brief Filter2D high radius */
    std::atomic<int> filter2d_n2{1};
    /*! \brief Algorithm to apply in hologram mode */
    std::atomic<SpaceTransformation> space_transformation{SpaceTransformation::None};
    /*! \brief Time transformation to apply in hologram mode */
    std::atomic<TimeTransformation> time_transformation{TimeTransformation::STFT};
    /*! \brief Number of images used by the time transformation */
    std::atomic<uint> time_transformation_size{1};
    /*! \brief Wave length of the laser */
    std::atomic<float> lambda{852e-9f};
    /*! \brief z value used by fresnel transform */
    std::atomic<float> zdistance{1.50f};
    /*! \brief Is convolution enabled */
    std::atomic<bool> convolution_enabled{false};
    /*! \brief Convolution type (file present in AppData) */
    // std::atomic<std::string> convolution_type{""};
    /*! \brief Is divide by convolution enabled */
    std::atomic<bool> divide_convolution_enabled{false};

    // View
    /*! \brief type of the image displayed */
    std::atomic<ImgType> img_type{ImgType::Modulus};
    // TODO: Add unwrap2d
    /*! \brief Are slices YZ and XZ enabled */
    std::atomic<bool> time_transformation_cuts_enabled{false};
    /*! \brief Is shift fft enabled (switching representation diagram) */
    std::atomic<bool> fft_shift_enabled{false};
    /*! \brief Is gpu lens display activated */
    std::atomic<bool> lens_view_enabled{false};
    /*! \brief Display the raw interferogram when we are in hologram mode. */
    std::atomic<bool> raw_view_enabled{false};

    XYView x{};
    XYView y{};
    PQView p{};
    PQView q{};

    /*! \brief Postprocessing renorm enabled */
    std::atomic<bool> renorm_enabled{true};
    /*! \brief Is the reticle overlay enabled */
    std::atomic<bool> reticle_view_enabled{false};
    /*! \brief Reticle border scale */
    std::atomic<float> reticle_scale{0.5f};

    /*! \brief Last window selected */
    std::atomic<WindowKind> current_window{WindowKind::XYview};
    WindowView* current = &xy;

    XY_XZ_YZ_WindowView xy{};
    XY_XZ_YZ_WindowView xz{};
    XY_XZ_YZ_WindowView yz{};
    WindowView filter2d{};

    // Composite images
    std::atomic<CompositeKind> composite_kind;
    std::atomic<bool> composite_auto_weights;
    Composite_RGB rgb{};
    Composite_RGB hsv{};

    // RGB //TODO: To replace
    std::atomic<uint> rgb_p_min{0};
    std::atomic<uint> rgb_p_max{0};
    std::atomic<float> weight_r{1};
    std::atomic<float> weight_g{1};
    std::atomic<float> weight_b{1};

    // HSV
    std::atomic<uint> composite_p_min_h{0};
    std::atomic<uint> composite_p_max_h{0};
    std::atomic<float> composite_slider_h_threshold_min{0.01f};
    std::atomic<float> composite_slider_h_threshold_max{1.0f};
    std::atomic<float> composite_low_h_threshold{0.2f};
    std::atomic<float> composite_high_h_threshold{99.8f};
    std::atomic<bool> h_blur_activated{false};
    std::atomic<uint> h_blur_kernel_size{1};

    std::atomic<bool> composite_p_activated_s{false};
    std::atomic<uint> composite_p_min_s{0};
    std::atomic<uint> composite_p_max_s{0};
    std::atomic<float> composite_slider_s_threshold_min{0.01f};
    std::atomic<float> composite_slider_s_threshold_max{1.0f};
    std::atomic<float> composite_low_s_threshold{0.2f};
    std::atomic<float> composite_high_s_threshold{99.8f};

    std::atomic<bool> composite_p_activated_v{false};
    std::atomic<uint> composite_p_min_v{0};
    std::atomic<uint> composite_p_max_v{0};
    std::atomic<float> composite_slider_v_threshold_min{0.01f};
    std::atomic<float> composite_slider_v_threshold_max{1.0f};
    std::atomic<float> composite_low_v_threshold{0.2f};
    std::atomic<float> composite_high_v_threshold{99.8f};

    // Advanced
    /*! \brief Max number of frames read each time by the thread_reader. */
    std::atomic<uint> file_buffer_size{32};
    /*! \brief Max size of input queue in number of images. */
    std::atomic<uint> input_buffer_size{256};
    /*! \brief Max size of frame record queue in number of images. */
    std::atomic<uint> record_buffer_size{64};
    /*! \brief Max size of output queue in number of images. */
    std::atomic<uint> output_buffer_size{64};
    /*! \brief Max size of time transformation cuts queue in number of images. */
    std::atomic<uint> time_transformation_cuts_output_buffer_size{8};
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
