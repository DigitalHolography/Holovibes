/*! \file
 *
 * \brief Implementation of the rendering features.
 */
#pragma once

#include <atomic>

#include "frame_desc.hh"
#include "function_vector.hh"
#include "queue.hh"
#include "rect.hh"
#include "shift_corners.cuh"
#include "global_state_holder.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::ImageType,                \
    holovibes::settings::XY,                       \
    holovibes::settings::XZ,                       \
    holovibes::settings::YZ,                       \
    holovibes::settings::Filter2d,                 \
    holovibes::settings::Filter2dViewEnabled,      \
    holovibes::settings::ChartDisplayEnabled,      \
    holovibes::settings::FftShiftEnabled,          \
    holovibes::settings::CutsViewEnabled,          \
    holovibes::settings::ReticleDisplayEnabled,    \
    holovibes::settings::ChartRecordEnabled,       \
    holovibes::settings::TimeTransformationSize,   \
    holovibes::settings::SignalZone,               \
    holovibes::settings::NoiseZone,                \
    holovibes::settings::ReticleZone

#define ONRESTART_SETTINGS                         \
    holovibes::settings::ContrastLowerThreshold,   \
    holovibes::settings::ContrastUpperThreshold,   \
    holovibes::settings::CutsContrastPOffset
    
#define ALL_SETTINGS REALTIME_SETTINGS, ONRESTART_SETTINGS

// clang-format on

namespace holovibes
{
class ICompute;
struct CoreBuffersEnv;
struct ChartEnv;
struct TimeTransformationEnv;
struct ImageAccEnv;
} // namespace holovibes

namespace holovibes::compute
{
using uint = unsigned int;

/*! \class Rendering
 *
 * \brief Manages rendering features and operations.
 *
 * This class handles various rendering operations, including FFT shifting, chart insertion, logarithmic
 * transformations, and contrast adjustments, leveraging CUDA for efficient computation.
 */
class Rendering
{
  public:
    /*! \brief Constructor to initialize the Rendering class with required settings and environments.
     *
     * \param fn_compute_vect Function vector for compute operations.
     * \param buffers Core buffer environment.
     * \param chart_env Chart environment for rendering charts.
     * \param image_acc_env Image accumulation environment.
     * \param time_transformation_env Time transformation environment.
     * \param input_fd Frame descriptor for input frames.
     * \param output_fd Frame descriptor for output frames.
     * \param stream CUDA stream for asynchronous operations.
     * \param settings Initialization settings.
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Rendering(FunctionVector& fn_compute_vect,
              const CoreBuffersEnv& buffers,
              ChartEnv& chart_env,
              const ImageAccEnv& image_acc_env,
              const TimeTransformationEnv& time_transformation_env,
              const camera::FrameDescriptor& input_fd,
              const camera::FrameDescriptor& output_fd,
              const cudaStream_t& stream,
              InitSettings settings)
        : fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , chart_env_(chart_env)
        , time_transformation_env_(time_transformation_env)
        , image_acc_env_(image_acc_env)
        , input_fd_(input_fd)
        , fd_(output_fd)
        , stream_(stream)
        , realtime_settings_(settings)
        , onrestart_settings_(settings)
    {
        // Hold 2 float values (min and max)
        cudaXMallocHost(&percent_min_max_, 2 * sizeof(float));
    }

    /*! \brief Destructor to free allocated resources. */
    ~Rendering();

    /*! \brief Inserts functions related to FFT shift into the function vector. */
    void insert_fft_shift();

    /*! \brief Inserts functions related to noise and signal charts into the function vector. */
    void insert_chart();

    /*! \brief Inserts functions related to logarithmic transformations into the function vector. */
    void insert_log();

    /*! \brief Inserts functions related to contrast adjustments into the function vector.
     *
     * \param autocontrast_request Atomic flag indicating if autocontrast is requested.
     * \param autocontrast_slice_xz_request Atomic flag indicating if autocontrast is requested for XZ slice.
     * \param autocontrast_slice_yz_request Atomic flag indicating if autocontrast is requested for YZ slice.
     * \param autocontrast_filter2d_request Atomic flag indicating if autocontrast is requested for 2D filter.
     */
    void insert_contrast(std::atomic<bool>& autocontrast_request,
                         std::atomic<bool>& autocontrast_slice_xz_request,
                         std::atomic<bool>& autocontrast_slice_yz_request,
                         std::atomic<bool>& autocontrast_filter2d_request);

    /*! \brief Updates a specific setting.
     *
     * \tparam T Type of the setting.
     * \param setting The setting to update.
     */
    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            spdlog::trace("[Rendering] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }

        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            spdlog::trace("[Rendering] [update_setting] {}", typeid(T).name());
            onrestart_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief Inserts the log10 transformation for the XY window. */
    void insert_main_log();

    /*! \brief Inserts the log10 transformation for the slices. */
    void insert_slice_log();

    /*! \brief Inserts the log10 transformation for the Filter2D view. */
    void insert_filter2d_view_log();

    /*! \brief Inserts the autocontrast computation. */
    void insert_compute_autocontrast(std::atomic<bool>& autocontrast_request,
                                     std::atomic<bool>& autocontrast_slice_xz_request,
                                     std::atomic<bool>& autocontrast_slice_yz_request,
                                     std::atomic<bool>& autocontrast_filter2d_request);

    /*! \brief Inserts the contrast application on a specific view.
     *
     * \param view The kind of window view for contrast application.
     */
    void insert_apply_contrast(WindowKind view);

    /*! \brief Calls autocontrast and sets the correct contrast variables.
     *
     * \param input Pointer to the input data.
     * \param width Width of the input data.
     * \param height Height of the input data.
     * \param offset Offset for the input data.
     * \param view The kind of window view for autocontrast.
     */
    void autocontrast_caller(float* input, const uint width, const uint height, const uint offset, WindowKind view);

    /*! \brief Retrieves a setting value.
     *
     * \tparam T Type of the setting.
     * \return The value of the setting.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            return realtime_settings_.get<T>().value;
        }

        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            return onrestart_settings_.get<T>().value;
        }
    }

    /*! \brief Vector of functions for compute operations. */
    FunctionVector& fn_compute_vect_;

    /*! \brief Core buffer environment. */
    const CoreBuffersEnv& buffers_;

    /*! \brief Chart environment for rendering charts. */
    ChartEnv& chart_env_;

    /*! \brief Time transformation environment. */
    const TimeTransformationEnv& time_transformation_env_;

    /*! \brief Image accumulation environment. */
    const ImageAccEnv& image_acc_env_;

    /*! \brief Descriptor for input frame size. */
    const camera::FrameDescriptor& input_fd_;

    /*! \brief Descriptor for output frame size. */
    const camera::FrameDescriptor& fd_;

    /*! \brief CUDA stream for asynchronous operations. */
    const cudaStream_t& stream_;

    /*! \brief Host memory for storing minimum and maximum percentage values. */
    float* percent_min_max_;

    /*! \brief Container for real-time settings. */
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;

    /*! \brief Container for settings applied on restart. */
    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
/*! \brief Checks if a setting exists in the Rendering class.
 *
 * \tparam T Type of the setting.
 */
template <typename T>
struct has_setting<T, compute::Rendering> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes