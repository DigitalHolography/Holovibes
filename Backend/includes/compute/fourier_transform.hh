/*! \file
 *
 * \brief Implementation of Fresnel Transform, Angular Spectrum and STFT algorithms.
 */
#pragma once

#include <algorithm>
#include <cufft.h>

#include "frame_desc.hh"
#include "rect.hh"
#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"
#include "function_vector.hh"
#include "logger.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

// Avoid conflict with std::max and std::min (min and max are defined in windows.h)
#undef max
#undef min

#pragma region Settings configuration
// clang-format off

#define PIPE_CYCLE_SETTINGS                        \
    holovibes::settings::X,                        \
    holovibes::settings::Y,                        \
    holovibes::settings::Q

#define PIPEREFRESH_SETTINGS                       \
    holovibes::settings::CutsViewEnabled,          \
    holovibes::settings::Filter2dEnabled,          \
    holovibes::settings::BatchSize,                \
    holovibes::settings::RecordMode,               \
    holovibes::settings::ImageType,                \
    holovibes::settings::P,                        \
    holovibes::settings::LensViewEnabled,          \
    holovibes::settings::XZ,                       \
    holovibes::settings::YZ,                       \
    holovibes::settings::Filter2dN1,               \
    holovibes::settings::Filter2dN2,               \
    holovibes::settings::Filter2dSmoothHigh,       \
    holovibes::settings::Filter2dSmoothLow,        \
    holovibes::settings::InputFilter,              \
    holovibes::settings::TimeTransformationSize,   \
    holovibes::settings::TimeTransformation,       \
    holovibes::settings::Lambda,                   \
    holovibes::settings::ZDistance,                \
    holovibes::settings::PixelSize,                \
    holovibes::settings::SpaceTransformation

#define ALL_SETTINGS PIPE_CYCLE_SETTINGS, PIPEREFRESH_SETTINGS

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
/*! \class FourierTransform
 *
 * \brief Class of Fourier Transform
 */
class FourierTransform
{
  public:
    /*! \brief Constructor */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    FourierTransform(std::shared_ptr<FunctionVector> fn_compute_vect,
                     const CoreBuffersEnv& buffers,
                     const camera::FrameDescriptor& fd,
                     cuda_tools::CufftHandle& spatial_transformation_plan,
                     TimeTransformationEnv& time_transformation_env,
                     MomentsEnv& moments_env,
                     const cudaStream_t& stream,
                     InitSettings settings)
        : gpu_lens_(nullptr)
        , lens_side_size_(std::max(fd.height, fd.width))
        , gpu_lens_queue_(nullptr)
        , fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , fd_(fd)
        , spatial_transformation_plan_(spatial_transformation_plan)
        , time_transformation_env_(time_transformation_env)
        , moments_env_(moments_env)
        , stream_(stream)
        , pipe_cycle_settings_(settings)
        , pipe_refresh_settings_(settings)
    {
        gpu_lens_.resize(fd_.get_frame_res());
    }

    /*! \brief enqueue functions relative to spatial fourier transforms. */
    void insert_fft(const uint width, const uint height);

    /*! \brief enqueue functions that store the p frame after the time transformation. */
    void insert_store_p_frame();

    /*! \brief Get Lens Queue used to display the Fresnel lens. */
    std::unique_ptr<Queue>& get_lens_queue();

    /*! \brief Initialize the Lens Queue. */
    void init_lens_queue();

    /*! \brief enqueue functions relative to temporal fourier transforms. */
    void insert_time_transform();

    /*!
     * \brief Enqueue the computations of the moments, after the stft
     *
     */
    void insert_moments();

    /**
     * \brief Splits 3 contiguous moments from a temporary buffer to their respective individual buffers.
     * Is used only when reading a moments file.
     */
    void insert_moments_split();

    /**
     * \brief Sends the respective moment to the output display (gpu_postprocess_frame)
     * if the corresponding image type is selected.
     *
     */
    void insert_moments_to_output();

    /*! \brief Enqueue functions relative to time transformation cuts display when there are activated */
    void insert_time_transformation_cuts_view(const camera::FrameDescriptor& fd,
                                              float* gpu_postprocess_frame_xz,
                                              float* gpu_postprocess_frame_yz);

    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting_v<T, decltype(pipe_cycle_settings_)>)
        {
            LOG_TRACE("[FourierTransform] [update_setting] {}", typeid(T).name());
            pipe_cycle_settings_.update_setting(setting);
        }

        if constexpr (has_setting_v<T, decltype(pipe_refresh_settings_)>)
        {
            LOG_TRACE("[FourierTransform] [update_setting] {}", typeid(T).name());
            pipe_refresh_settings_.update_setting(setting);
        }
    }

    /*! \brief Update the realtime settings */
    inline void pipe_cycle_apply_updates() { pipe_cycle_settings_.apply_updates(); }

    /*! \brief Update the pipe refresh settings */
    inline void pipe_refresh_apply_updates() { pipe_refresh_settings_.apply_updates(); }

  private:
    /*! \brief Enqueue the call to filter2d cuda function. */
    void insert_filter2d();

    /*! \brief Compute lens and enqueue the call to the fresnel_transform cuda function. */
    void insert_fresnel_transform();

    /*! \brief Compute lens and enqueue the call to the angular_spectrum cuda function. */
    void insert_angular_spectrum(bool filter2d_enabled);

    /*! \brief Enqueue the Fresnel lens into the Lens Queue.
     *
     * It will enqueue the lens, and normalize it, in order to display it correctly later.
     */
    void enqueue_lens(SpaceTransformation space_transformation);

    /*! \brief Enqueue stft time filtering. */
    void insert_stft();

    /*! \brief Enqueue functions relative to filtering using diagonalization and eigen values.
     *
     * This should eventually replace stft
     */
    void insert_pca();

    void insert_ssa_stft();

    /**
     * @brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting_v<T, decltype(pipe_cycle_settings_)>)
            return pipe_cycle_settings_.get<T>().value;

        if constexpr (has_setting_v<T, decltype(pipe_refresh_settings_)>)
            return pipe_refresh_settings_.get<T>().value;
    }

    /*! \brief Roi zone of Filter 2D */
    units::RectFd filter2d_zone_;
    units::RectFd filter2d_subzone_;

    /*! \brief Lens used for fresnel transform (During Fresnel Transform itself and Angular Spectrum) */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_lens_;
    /*! \brief Size of a size of the lens (lens is always a square) */
    uint lens_side_size_ = {0};
    /*! \brief Lens Queue. Used for displaying the lens. */
    std::unique_ptr<Queue> gpu_lens_queue_;

    /*! \brief Size of the buffer needed by cusolver for internal use */
    int cusolver_work_buffer_size_;
    /*! \brief Buffer needed by cusolver for internal use */
    cuda_tools::CudaUniquePtr<cuComplex> cusolver_work_buffer_;

    /*! \brief Vector function in which we insert the processing */
    std::shared_ptr<FunctionVector> fn_compute_vect_;
    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;
    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Pland 2D. Used by FFTs (1, 2, filter2D). */
    cuda_tools::CufftHandle& spatial_transformation_plan_;
    /*! \brief Time transformation environment. */
    TimeTransformationEnv& time_transformation_env_;
    /*! \brief Moments environment. */
    MomentsEnv& moments_env_;
    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    DelayedSettingsContainer<PIPE_CYCLE_SETTINGS> pipe_cycle_settings_;
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
