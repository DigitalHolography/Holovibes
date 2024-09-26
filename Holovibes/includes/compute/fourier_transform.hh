/*! \file
 *
 * \brief Implementation of FFT1, FFT2 and STFT algorithms.
 */
#pragma once

#include <cufft.h>

#include "frame_desc.hh"
#include "rect.hh"
#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"
#include "function_vector.hh"
#include "global_state_holder.hh"
#include "logger.hh"

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
} // namespace holovibes

namespace holovibes::compute
{
/*! \class FourierTransform
 *
 * \brief #TODO Add a description for this class
 */
class FourierTransform
{
  public:
    /*! \brief Constructor */
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

    /*! \brief enqueue functions relative to spatial fourier transforms. */
    void insert_fft(float* gpu_filter2d_mask, const uint width, const uint height);

    /*! \brief enqueue functions that store the p frame after the time transformation. */
    void insert_store_p_frame();

    /*! \brief Get Lens Queue used to display the Fresnel lens. */
    std::unique_ptr<Queue>& get_lens_queue();

    /*! \brief enqueue functions relative to temporal fourier transforms. */
    void insert_time_transform();

    /*! \brief Enqueue functions relative to time transformation cuts display when there are activated */
    void insert_time_transformation_cuts_view(const camera::FrameDescriptor& fd,
                                              float* gpu_postprocess_frame_xz,
                                              float* gpu_postprocess_frame_yz);

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
    /*! \brief Enqueue the call to filter2d cuda function. */
    void insert_filter2d();

    /*! \brief Compute lens and enqueue the call to fft1 cuda function. */
    void insert_fft1();

    /*! \brief Compute lens and enqueue the call to fft2 cuda function. */
    void insert_fft2(bool filter2d_enabled);

    /*! \brief Enqueue the Fresnel lens into the Lens Queue.
     *
     * It will enqueue the lens, and normalize it, in order to display it correctly later.
     */
    void enqueue_lens(SpaceTransformation space_transformation);

    /*! \brief Enqueue stft time filtering. */
    void insert_stft();

    /*!
     * \brief Returns the Discrete Fourier Transform sample frequencies.
     * The returned float array contains the frequency bin centers in cycles times unit of the sample spacing (with zero
     * at the start).
     * For instance, if the sample spacing is in seconds, then the frequency unit is cycles/second.
     * In our case, we reason in terms of sampling rate (fps), which is the inverse of the sample spacing ; this doesn't
     * affect the frequency unit.
     *
     * For a given sampling rate (input_fps) Fs, and a window length n (time_transformation_size), the sample
     * frequencies correspond to :
     *
     * f = [0, 1, ...,   n/2-1,     -n/2, ..., -1] * fs / n   if n is even
     * f = [0, 1, ..., (n - 1) / 2, -(n - 1) / 2, ..., -1] * fs / n if n is odd
     *
     * The functions compute f0, f1 and f2, corresponding to f at order 0 (an array of size time_transformation_size)
     * filled with 1, f at order 1, and f at order 2 (f^2)
     */
    void ICompute::fft_freqs();

    void insert_moments();

    /*! \brief Enqueue functions relative to filtering using diagonalization and eigen values.
     *
     * This should eventually replace stft
     */
    void insert_pca();

    void insert_ssa_stft(ViewPQ view_q);

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

    /*! \brief Roi zone of Filter 2D */
    units::RectFd filter2d_zone_;
    units::RectFd filter2d_subzone_;

    /*! \brief Lens used for fresnel transform (During FFT1 and FFT2) */
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
    FunctionVector& fn_compute_vect_;
    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;
    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Pland 2D. Used by FFTs (1, 2, filter2D). */
    cuda_tools::CufftHandle& spatial_transformation_plan_;
    /*! \brief Time transformation environment. */
    TimeTransformationEnv& time_transformation_env_;
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
