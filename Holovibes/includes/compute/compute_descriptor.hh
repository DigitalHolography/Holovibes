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

// struct
#include "composite_struct.hh"
#include "view_struct.hh"

namespace holovibes
{

#define __HOLOVIBES_VERSION__ "10.4"
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

    inline unsigned get_renorm_constant() const { return renorm_constant; }
    inline void set_renorm_constant(unsigned renorm_constant) { this->renorm_constant = renorm_constant; }

    inline int get_filter2d_smooth_low() const { return filter2d_smooth_low; }
    inline void set_filter2d_smooth_low(int filter2d_smooth_low) { this->filter2d_smooth_low = filter2d_smooth_low; }

    inline int get_filter2d_smooth_high() const { return filter2d_smooth_high; }
    inline void set_filter2d_smooth_high(int filter2d_smooth_high)
    {
        this->filter2d_smooth_high = filter2d_smooth_high;
    }

    inline bool get_reticle_display_enabled() const { return reticle_display_enabled; }
    inline void set_reticle_display_enabled(bool reticle_enabled) { this->reticle_display_enabled = reticle_enabled; }

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

    /*! \brief Is the reticle overlay enabled */
    std::atomic<bool> reticle_display_enabled{false};

    // Advanced
    /*! \brief Filter2D low smoothing */ // May be moved in filter2d Struct
    std::atomic<int> filter2d_smooth_low{0};
    /*! \brief Filter2D high smoothing */
    std::atomic<int> filter2d_smooth_high{0};
    std::atomic<float> contrast_lower_threshold{0.5f};
    std::atomic<float> contrast_upper_threshold{99.5f};
    /*! \brief postprocessing remormalize multiplication constant */
    std::atomic<unsigned> renorm_constant{5};
    std::atomic<uint> cuts_contrast_p_offset{2};

#pragma endregion
};
} // namespace holovibes
