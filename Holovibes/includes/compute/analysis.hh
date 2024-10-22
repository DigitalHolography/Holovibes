/*! \file
 *
 * \brief
 */
#pragma once

#include "frame_desc.hh"
#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"
#include "function_vector.hh"
#include "logger.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::RecordMode,               \
    holovibes::settings::ImageType,                \
    holovibes::settings::X,                        \
    holovibes::settings::Y,                        \
    holovibes::settings::P,                        \
    holovibes::settings::Q,                        \
    holovibes::settings::Filter2dEnabled,          \
    holovibes::settings::CutsViewEnabled,          \
    holovibes::settings::TimeTransformationSize,   \
    holovibes::settings::TimeTransformation,       \
    holovibes::settings::Lambda,                   \
    holovibes::settings::ZDistance,                \
    holovibes::settings::PixelSize,                \
    holovibes::settings::Filter2dN1,               \
    holovibes::settings::Filter2dN2,               \
    holovibes::settings::Filter2dSmoothHigh,       \
    holovibes::settings::Filter2dSmoothLow,        \
    holovibes::settings::SpaceTransformation

#define PIPEREFRESH_SETTINGS                         \
    holovibes::settings::BatchSize,                  \
    holovibes::settings::XZ,                       \
    holovibes::settings::YZ,                       \
    holovibes::settings::InputFilter,                \
    holovibes::settings::FilterEnabled

#define ALL_SETTINGS REALTIME_SETTINGS, PIPEREFRESH_SETTINGS

// clang-format on

namespace holovibes
{
class Queue;
struct BatchEnv;
struct TimeTransformationEnv;
struct CoreBuffersEnv;
struct MomentsEnv;
} // namespace holovibes

namespace holovibes::compute
{
/*! \class Analysis
 *
 * \brief #TODO Add a description for this class
 */
class Analysis
{
  public:
    /*! \brief Constructor */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Analysis(FunctionVector& fn_compute_vect,
             const CoreBuffersEnv& buffers,
             const camera::FrameDescriptor& fd,
             MomentsEnv& moments_env,
             const cudaStream_t& stream,
             InitSettings settings)
        :
        , fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , fd_(fd)
        , moments_env_(moments_env)
        , stream_(stream)
        , realtime_settings_(settings)
        , pipe_refresh_settings_(settings)
    {
    }

    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            LOG_TRACE("[FourierTransform] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
        if constexpr (has_setting<T, decltype(pipe_refresh_settings_)>::value)
        {
            LOG_TRACE("[FourierTransform] [update_setting] {}", typeid(T).name());
            pipe_refresh_settings_.update_setting(setting);
        }
    }

    inline void pipe_refresh_apply_updates() { pipe_refresh_settings_.apply_updates(); }

  private:
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

        if constexpr (has_setting<T, decltype(pipe_refresh_settings_)>::value)
        {
            return pipe_refresh_settings_.get<T>().value;
        }
    }

    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;
    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;
    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Moments environment. */
    MomentsEnv& moments_env_;
    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
    DelayedSettingsContainer<PIPEREFRESH_SETTINGS> pipe_refresh_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::FourierTransform> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
