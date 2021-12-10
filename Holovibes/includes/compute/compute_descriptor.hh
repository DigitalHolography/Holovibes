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
    /*! \} */

#pragma region Atomics vars

#pragma endregion
};
} // namespace holovibes
