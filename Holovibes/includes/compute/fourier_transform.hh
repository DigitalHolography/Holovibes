/*! \file
 *
 * \brief Implementation of FFT1, FFT2, and STFT algorithms.
 */
#pragma once

#include <cufft.h>

#include "frame_desc.hh"
#include "rect.hh"
#include "cuda_tools/unique_ptr.hh"
#include "cuda_tools/array.hh"
#include "cuda_tools/cufft_handle.hh"
#include "function_vector.hh"
#include "global_state_holder.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
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

#define PIPEREFRESH_SETTINGS                       \
    holovibes::settings::BatchSize,                \
    holovibes::settings::XZ,                       \
    holovibes::settings::YZ,                       \
    holovibes::settings::InputFilter,              \
    holovibes::settings::FilterEnabled

#define ALL_SETTINGS REALTIME_SETTINGS, PIPEREFRESH_SETTINGS

// clang-format on

namespace holovibes
{
class Queue;
struct BatchEnv;
struct TimeTransformationEnv;
struct CoreBuffersEnv;
} // namespace holovibes

namespace holovibes::compute
{
/*! \class FourierTransform
 *
 * \brief Manages the execution of FFT1, FFT2, and STFT algorithms.
 *
 * This class handles the initialization, execution, and management of various Fourier transform algorithms,
 * including FFT1, FFT2, and Short-Time Fourier Transform (STFT), leveraging CUDA and cuFFT libraries.
 */
class FourierTransform
{
  public:
    /*! \brief Constructor to initialize the FourierTransform class with required settings and environments.
     *
     * \param fn_compute_vect Function vector for compute operations.
     * \param buffers Core buffer environment.
     * \param fd Frame descriptor for input frames.
     * \param spatial_transformation_plan CUFFT handle for spatial transformations.
     * \param time_transformation_env Time transformation environment.
     * \param stream CUDA stream for asynchronous operations.
     * \param settings Initialization settings.
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    FourierTransform(FunctionVector& fn_compute_vect,
                     const CoreBuffersEnv& buffers,
                     const camera::FrameDescriptor& fd,
                     cuda_tools::CufftHandle& spatial_transformation_plan,
                     TimeTransformationEnv& time_transformation_env,
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
        , stream_(stream)
        , realtime_settings_(settings)
        , pipe_refresh_settings_(settings)
    {
        gpu_lens_.resize(fd_.get_frame_res());
    }

    /*! \brief Enqueues functions related to spatial Fourier transforms.
     *
     * \param gpu_filter2d_mask GPU buffer for the 2D filter mask.
     * \param width Width of the input data.
     * \param height Height of the input data.
     */
    void insert_fft(float* gpu_filter2d_mask, const uint width, const uint height);

    /*! \brief Enqueues functions that store the p frame after the time transformation. */
    void insert_store_p_frame();

    /*! \brief Retrieves the Lens Queue used to display the Fresnel lens.
     *
     * \return A unique pointer to the Lens Queue.
     */
    std::unique_ptr<Queue>& get_lens_queue();

    /*! \brief Enqueues functions related to temporal Fourier transforms. */
    void insert_time_transform();

    /*! \brief Enqueues functions related to displaying time transformation cuts.
     *
     * \param fd Frame descriptor for the cuts.
     * \param gpu_postprocess_frame_xz GPU buffer for the XZ cut.
     * \param gpu_postprocess_frame_yz GPU buffer for the YZ cut.
     */
    void insert_time_transformation_cuts_view(const camera::FrameDescriptor& fd,
                                              float* gpu_postprocess_frame_xz,
                                              float* gpu_postprocess_frame_yz);

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
            spdlog::trace("[FourierTransform] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
        if constexpr (has_setting<T, decltype(pipe_refresh_settings_)>::value)
        {
            spdlog::trace("[FourierTransform] [update_setting] {}", typeid(T).name());
            pipe_refresh_settings_.update_setting(setting);
        }
    }

    /*! \brief Applies updates to the pipe refresh settings. */
    inline void pipe_refresh_apply_updates()
    {
        pipe_refresh_settings_.apply_updates();
    }

  private:
    /*! \brief Enqueues the call to the 2D filter CUDA function. */
    void insert_filter2d();

    /*! \brief Computes the lens and enqueues the call to the FFT1 CUDA function. */
    void insert_fft1();

    /*! \brief Computes the lens and enqueues the call to the FFT2 CUDA function.
     *
     * \param filter2d_enabled Indicates if the 2D filter is enabled.
     */
    void insert_fft2(bool filter2d_enabled);

    /*! \brief Enqueues the Fresnel lens into the Lens Queue.
     *
     * Normalizes and enqueues the lens for correct display.
     *
     * \param space_transformation Space transformation to apply.
     */
    void enqueue_lens(SpaceTransformation space_transformation);

    /*! \brief Enqueues STFT time filtering. */
    void insert_stft();

    /*! \brief Enqueues functions related to filtering using diagonalization and eigenvalues.
     *
     * This method is intended to eventually replace the STFT method.
     */
    void insert_pca();

    /*! \brief Enqueues SSA-STFT processing.
     *
     * \param view_q View parameters for the processing.
     */
    void insert_ssa_stft(ViewPQ view_q);

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

        if constexpr (has_setting<T, decltype(pipe_refresh_settings_)>::value)
        {
            return pipe_refresh_settings_.get<T>().value;
        }
    }

    /*! \brief Region of interest for the 2D filter. */
    units::RectFd filter2d_zone_;
    /*! \brief Sub-region of interest for the 2D filter. */
    units::RectFd filter2d_subzone_;

    /*! \brief Lens used for Fresnel transform during FFT1 and FFT2. */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_lens_;
    /*! \brief Size of one side of the lens (lens is always a square). */
    uint lens_side_size_ = {0};
    /*! \brief Lens Queue used for displaying the lens. */
    std::unique_ptr<Queue> gpu_lens_queue_;

    /*! \brief Size of the buffer needed by cuSolver for internal use. */
    int cusolver_work_buffer_size_;
    /*! \brief Buffer needed by cuSolver for internal use. */
    cuda_tools::CudaUniquePtr<cuComplex> cusolver_work_buffer_;

    /*! \brief Vector of functions for compute operations. */
    FunctionVector& fn_compute_vect_;
    /*! \brief Core buffer environment. */
    const CoreBuffersEnv& buffers_;
    /*! \brief Descriptor for frame size. */
    const camera::FrameDescriptor& fd_;
    /*! \brief CUFFT handle for 2D transformations. */
    cuda_tools::CufftHandle& spatial_transformation_plan_;
    /*! \brief Time transformation environment. */
    TimeTransformationEnv& time_transformation_env_;
    /*! \brief CUDA stream for asynchronous operations. */
    const cudaStream_t& stream_;

    /*! \brief Container for real-time settings. */
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
    /*! \brief Container for delayed pipe refresh settings. */
    DelayedSettingsContainer<PIPEREFRESH_SETTINGS> pipe_refresh_settings_;
};
} // namespace holovibes

namespace holovibes
{
/*! \brief Checks if a setting exists in the FourierTransform class.
 *
 * \tparam T Type of the setting.
 */
template <typename T>
struct has_setting<T, compute::FourierTransform> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes