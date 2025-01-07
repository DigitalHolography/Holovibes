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
#include "apply_mask.cuh"
#include "logger.hh"
#include "convolution.cuh"
#include <cufft.h>
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
 * \brief Class of the rendering features
 */
class Rendering
{
  public:
    /*! \brief Constructor */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Rendering(std::shared_ptr<FunctionVector> fn_compute_vect,
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
    ~Rendering();

    /*! \brief insert the functions relative to the fft shift. */
    void insert_fft_shift();
    /*! \brief insert the functions relative to noise and signal chart. */
    void insert_chart();
    /*! \brief insert the functions relative to the log10. */
    void insert_log();
    /*! \brief insert the functions relative to the contrast. */
    void insert_contrast();

    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            LOG_TRACE("[Rendering] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }

        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            LOG_TRACE("[Rendering] [update_setting] {}", typeid(T).name());
            onrestart_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief insert the log10 on the XY window */
    void insert_main_log();
    /*! \brief insert the log10 on the slices */
    void insert_slice_log();
    /*! \brief insert the log10 on the Filter2D view */
    void insert_filter2d_view_log();

    /*! \brief insert the autocontrast computation */
    void insert_compute_autocontrast();

    /*! \brief Check whether autocontrast should be applied or not for each view */
    void request_autocontrast();

    /*! \brief insert the constrast on a view */
    void insert_apply_contrast(WindowKind view);

    /*! \brief Calls autocontrast and set the correct contrast variables */
    void autocontrast_caller(float* input, const uint width, const uint height, const uint offset, WindowKind view);

    /*! \brief Tell if the contrast should be applied
     *
     * \param request[in] The request for autocontrast
     * \param queue[in] The accumulation queue
     * \return true if the contrast should be applied
     */
    inline bool should_apply_contrast(bool request, const std::unique_ptr<Queue>& queue)
    {
        if (!request)
            return false;

        // Apply contrast if there is no queue = accumulation set to 1
        if (!queue)
            return true;

        // Else there are frames in the accumulutation queue. We calculate autocontrast on the first frame to calibrate
        // the contrast and apply it one more time when the queue is full to fine tune it. It's done to reduce the
        // blinking effect when the contrast is applied.
        return queue->is_full() || queue->get_size() == 1;
    }

    /**
     * @brief Helper function to get a settings value.
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

    /*! \brief Vector function in which we insert the processing */
    std::shared_ptr<FunctionVector> fn_compute_vect_;
    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;
    /*! \brief Chart variables */
    ChartEnv& chart_env_;
    /*! \brief Time transformation environment */
    const TimeTransformationEnv& time_transformation_env_;
    /*! \brief Image accumulation environment */
    const ImageAccEnv& image_acc_env_;
    /*! \brief Describes the input frame size */
    const camera::FrameDescriptor& input_fd_;
    /*! \brief Describes the output frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    float* percent_min_max_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;

    bool autocontrast_xy_ = false;
    bool autocontrast_xz_ = false;
    bool autocontrast_yz_ = false;
    bool autocontrast_filter2d_ = false;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::Rendering> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
