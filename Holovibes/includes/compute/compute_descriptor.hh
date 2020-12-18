/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file
 *
 * Contains compute parameters. */
#pragma once

#include <atomic>
#include <mutex>
#include "observable.hh"
#include "rect.hh"

// enum
#include "enum_space_transformation.hh"
#include "enum_square_input_mode.hh"
#include "enum_time_transformation.hh"
#include "enum_computation.hh"
#include "enum_img_type.hh"
#include "enum_filter_2d.hh"
#include "enum_access_mode.hh"
#include "enum_window_kind.hh"
#include "enum_composite_kind.hh"

namespace holovibes
{
/*!< Current version of this project. */
const static std::string version = "v8.9.4";

/*! \brief Contains compute parameters.
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

    /*! \brief	The position of the point used to obtain XZ and YZ views */
    units::PointFd stft_slice_cursor;

    /*! \brief	The zone for the signal chart*/
    units::RectFd signal_zone;
    /*! \brief	The zone for the noise chart */
    units::RectFd noise_zone;
    /*! \brief	Limits the computation to only this zone. Also called Filter
     * 2D*/
    units::RectFd stft_roi_zone;
    /*! \brief	The subzone of the filter2D area in band-pass mode */
    units::RectFd filter2D_sub_zone;
    /*! \brief	The area on which we'll normalize the colors*/
    units::RectFd composite_zone;
    /*! \brief  The area used to limit the stft computations. */
    units::RectFd zoomed_zone;
    /*! \brief	The zone of the reticle area */
    units::RectFd reticle_zone;

  public:
    /*! \brief ComputeDescriptor constructor
     * Initialize the compute descriptor to default values of computation. */
    ComputeDescriptor();

    /*! \brief ComputeDescriptor destructor. */
    ~ComputeDescriptor();

    /*! \brief Assignment operator
     * The assignment operator is explicitely defined because std::atomic type
     * does not allow to generate assignments operator automatically. */
    ComputeDescriptor& operator=(const ComputeDescriptor& cd);

    /*!
     * @{
     *
     * \brief	Accessor to the selected zone
     *
     * \param			rect	The rectangle to process
     * \param 		  	m   	An AccessMode to process.
     */

    void signalZone(units::RectFd& rect, AccessMode m);
    void noiseZone(units::RectFd& rect, AccessMode m);
    //! @}

    /*!
     * @{
     *
     * \brief	Getter of the overlay positions.
     *
     */

    units::RectFd getStftZone() const;
    units::RectFd getFilter2DSubZone() const;
    units::RectFd getCompositeZone() const;
    units::RectFd getZoomedZone() const;
    units::PointFd getStftCursor() const;
    units::RectFd getReticleZone() const;
    //! @}

    /*!
     * @{
     *
     * \brief	Setter of the overlay positions.
     *
     */
    void setStftZone(const units::RectFd& rect);
    void setFilter2DSubZone(const units::RectFd& rect);
    void setCompositeZone(const units::RectFd& rect);
    void setZoomedZone(const units::RectFd& rect);
    void setStftCursor(const units::PointFd& rect);
    void setReticleZone(const units::RectFd& rect);
    //! @}

    /*!
     * @{
     *
     * \brief General getters / setters to avoid code duplication
     *
     */
    float get_contrast_min(WindowKind kind) const;
    float get_contrast_max(WindowKind kind) const;

    // Qt rounds the value by default.
    // In order to compare the compute descriptor values
    // these values also needs to be round.
    float get_truncate_contrast_min(WindowKind kind,
                                    const int precision = 2) const;
    float get_truncate_contrast_max(WindowKind kind,
                                    const int precision = 2) const;

    bool get_img_log_scale_slice_enabled(WindowKind kind) const;
    bool get_img_acc_slice_enabled(WindowKind kind) const;
    unsigned get_img_acc_slice_level(WindowKind kind) const;

    void set_contrast_min(WindowKind kind, float value);
    void set_contrast_max(WindowKind kind, float value);
    void set_log_scale_slice_enabled(WindowKind kind, bool value);
    void set_accumulation(WindowKind kind, bool value);
    void set_accumulation_level(WindowKind kind, float value);
    //! @}

#pragma region Atomics vars
    //! Is the computation stopped
    std::atomic<bool> is_computation_stopped{true};
    //! Mode of computation of the image
    std::atomic<Computation> compute_mode{Computation::Raw};
    //! Algorithm to apply in hologram mode
    std::atomic<SpaceTransformation> space_transformation{
        SpaceTransformation::None};
    //! Time transformation to apply in hologram mode
    std::atomic<TimeTransformation> time_transformation{
        TimeTransformation::STFT};
    //! Square conversion mode of the input
    std::atomic<SquareInputMode> square_input_mode{
        SquareInputMode::NO_MODIFICATION};
    //! type of the image displayed
    std::atomic<ImgType> img_type{ImgType::Modulus};

    //! Last window selected
    std::atomic<WindowKind> current_window{WindowKind::XYview};
    /*! \brief Number of images dequeued from input to gpu_input_queue and batch
     *size of space transformation
     **
     ** time_transformation_stride is decorelated from batch size for
     *performances reasons
     */
    std::atomic<ushort> batch_size{1};
    //! Number of images used by SFTF i.e. depth of the SFTF cube
    std::atomic<ushort> time_transformation_size{1};
    //! index in the depth axis
    std::atomic<ushort> pindex{0};

    //! wave length of the laser
    std::atomic<float> lambda{532e-9f};
    //! Input matrix used for convolution
    std::vector<float> convo_matrix;
    //! z value used by fresnel transform
    std::atomic<float> zdistance{1.50f};

    //! minimum constrast value in xy view
    std::atomic<float> contrast_min_slice_xy{1.f};
    //! maximum constrast value in xy view
    std::atomic<float> contrast_max_slice_xy{65535.f};
    //! minimum constrast value in xz view
    std::atomic<float> contrast_min_slice_xz{1.f};
    //! maximum constrast value in xz view
    std::atomic<float> contrast_max_slice_xz{65535.f};
    //! minimum constrast value in yz view
    std::atomic<float> contrast_min_slice_yz{1.f};
    //! maximum constrast value in yz view
    std::atomic<float> contrast_max_slice_yz{65535.f};
    //! invert contrast
    std::atomic<bool> contrast_invert{false};

    std::atomic<float> contrast_lower_threshold{0.5f};

    std::atomic<float> contrast_upper_threshold{99.5f};

    std::atomic<ushort> cuts_contrast_p_offset{2};
    //! Size of a pixel in micron
    std::atomic<float> pixel_size;
    //! Width of the matrix used for convolution
    std::atomic<uint> convo_matrix_width{0};
    //! Height of the matrix used for convolution
    std::atomic<uint> convo_matrix_height{0};
    //! Z of the matrix used for convolution
    std::atomic<uint> convo_matrix_z{0};
    //! Number of pipe iterations between two temporal demodulation.
    std::atomic<uint> time_transformation_stride{1};

    std::atomic<int> unwrap_history_size{1};
    //! is convolution enabled
    std::atomic<bool> convolution_enabled{false};
    //! is divide by convolution enabled
    std::atomic<bool> divide_convolution_enabled{false};
    //! postprocessing renorm enabled
    std::atomic<bool> renorm_enabled{true};
    //! postprocessing remormalize multiplication constant
    std::atomic<unsigned> renorm_constant{5};
    //! is log scale in slice XY enabled
    std::atomic<bool> log_scale_slice_xy_enabled{false};
    //! is log scale in slice XZ enabled
    std::atomic<bool> log_scale_slice_xz_enabled{false};
    //! is log scale in slice YZ enabled
    std::atomic<bool> log_scale_slice_yz_enabled{false};
    //! is shift fft enabled (switching representation diagram)
    std::atomic<bool> fft_shift_enabled{false};
    //! enables the contrast for the slice xy, yz and xz
    std::atomic<bool> contrast_enabled{false};
    //! enables auto refresh of the contrast
    std::atomic<bool> contrast_auto_refresh{true};
    //! type of filter 2D
    std::atomic<Filter2DType> filter_2d_type{Filter2DType::None};

    //! are slices YZ and XZ enabled
    std::atomic<bool> time_transformation_cuts_enabled{false};
    //! is gpu lens display activated
    std::atomic<bool> gpu_lens_display_enabled{false};
    //! enables the signal and noise chart display
    std::atomic<bool> chart_display_enabled{false};
    //! enables the signal and noise chart record
    std::atomic<bool> chart_record_enabled{false};

    //! Number of frame per seconds displayed
    std::atomic<float> display_rate{30};

    //! is img average in view XY enabled (average of output over time, i.e.
    //! phase compensation)
    std::atomic<bool> img_acc_slice_xy_enabled{false};
    //! is img average in view XZ enabled
    std::atomic<bool> img_acc_slice_xz_enabled{false};
    //! is img average in view YZ enabled
    std::atomic<bool> img_acc_slice_yz_enabled{false};
    //! number of image in view XY to average
    std::atomic<uint> img_acc_slice_xy_level{1};
    //! number of image in view XZ to average
    std::atomic<uint> img_acc_slice_xz_level{1};
    //! number of image in view YZ to average
    std::atomic<uint> img_acc_slice_yz_level{1};

    //! is p average enabled (average image over multiple depth index)
    std::atomic<bool> p_accu_enabled{false};
    //! difference between p min and p max
    std::atomic<short> p_acc_level{1};

    //! is x average in view YZ enabled (average of columns between both
    //! selected columns)
    std::atomic<bool> x_accu_enabled{false};
    //! difference between x min and x max
    std::atomic<short> x_acc_level{1};

    //! is y average in view XZ enabled (average of lines between both selected
    //! lines)
    std::atomic<bool> y_accu_enabled{false};
    //! difference between y min and y max
    std::atomic<short> y_acc_level{1};

    //! Display the raw interferogram when we are in hologram mode.
    std::atomic<bool> raw_view_enabled{false};

    //! Wait the beginning of the file to start the recording.
    std::atomic<bool> synchronized_record{false};

    //! Is the reticle overlay enabled
    std::atomic<bool> reticle_enabled{false};
    //! Reticle border scale.
    std::atomic<float> reticle_scale{0.5f};

    //! Number of bits to shift when in raw mode
    std::atomic<ushort> raw_bitshift{0};

    //! Composite images
    //! \{

    //! RGB
    std::atomic<ushort> composite_p_red{0};
    std::atomic<ushort> composite_p_blue{0};
    std::atomic<float> weight_r{1};
    std::atomic<float> weight_g{1};
    std::atomic<float> weight_b{1};

    //! HSV
    std::atomic<ushort> composite_p_min_h{0};
    std::atomic<ushort> composite_p_max_h{0};
    std::atomic<float> slider_h_threshold_min{0.01f};
    std::atomic<float> slider_h_threshold_max{1.0f};
    std::atomic<float> composite_low_h_threshold{0.2f};
    std::atomic<float> composite_high_h_threshold{99.8f};
    std::atomic<bool> h_blur_activated{false};
    std::atomic<uint> h_blur_kernel_size{1};

    std::atomic<bool> composite_p_activated_s{false};
    std::atomic<ushort> composite_p_min_s{0};
    std::atomic<ushort> composite_p_max_s{0};
    std::atomic<float> slider_s_threshold_min{0.01f};
    std::atomic<float> slider_s_threshold_max{1.0f};
    std::atomic<float> composite_low_s_threshold{0.2f};
    std::atomic<float> composite_high_s_threshold{99.8f};

    std::atomic<bool> composite_p_activated_v{false};
    std::atomic<ushort> composite_p_min_v{0};
    std::atomic<ushort> composite_p_max_v{0};
    std::atomic<float> slider_v_threshold_min{0.01f};
    std::atomic<float> slider_v_threshold_max{1.0f};
    std::atomic<float> composite_low_v_threshold{0.2f};
    std::atomic<float> composite_high_v_threshold{99.8f};

    std::atomic<CompositeKind> composite_kind;

    std::atomic<bool> composite_auto_weights_;

    /*! \brief Flag to signal the recording has started/stopped. Used by the
    ** thread compute and set by thread recorder
    */
    std::atomic<bool> frame_record_enabled{false};
    //! \}

#pragma endregion
};
} // namespace holovibes
