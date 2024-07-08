/*! \file
 *
 * \brief Implementation of postprocessing features on complex buffers.
 */
#pragma once

#include <vector>

#include "function_vector.hh"
#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "cufft_handle.hh"
#include "global_state_holder.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::ImageType,                \
    holovibes::settings::RenormEnabled,            \
    holovibes::settings::ConvolutionMatrix,        \
    settings::ConvolutionEnabled,                  \
    settings::DivideConvolutionEnabled

#define ONRESTART_SETTINGS                         \
    holovibes::settings::RenormConstant

#define ALL_SETTINGS REALTIME_SETTINGS, ONRESTART_SETTINGS

// clang-format on

using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
struct CoreBuffersEnv;
} // namespace holovibes

namespace holovibes::compute
{
/*! \class Postprocessing
 *
 * \brief Manages postprocessing features for complex buffers.
 *
 * This class handles various postprocessing operations on complex buffers, including convolution
 * and renormalization, leveraging CUDA and cuFFT libraries for efficient computation.
 */
class Postprocessing
{
  public:
    /*! \brief Constructor to initialize the Postprocessing class with required settings and environments.
     *
     * \param fn_compute_vect Function vector for compute operations.
     * \param buffers Core buffer environment.
     * \param input_fd Frame descriptor for input frames.
     * \param stream CUDA stream for asynchronous operations.
     * \param settings Initialization settings.
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Postprocessing(FunctionVector& fn_compute_vect,
                   CoreBuffersEnv& buffers,
                   const camera::FrameDescriptor& input_fd,
                   const cudaStream_t& stream,
                   InitSettings settings)
        : gpu_kernel_buffer_()
        , cuComplex_buffer_()
        , hsv_arr_()
        , reduce_result_(1) // allocate a unique double
        , fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , fd_(input_fd)
        , convolution_plan_(input_fd.height, input_fd.width, CUFFT_C2C)
        , stream_(stream)
        , realtime_settings_(settings)
        , onrestart_settings_(settings)
    {
    }

    /*! \brief Initializes convolution by allocating the corresponding buffer. */
    void init();

    /*! \brief Frees the resources allocated for postprocessing. */
    void dispose();

    /*! \brief Inserts the convolution function into the function vector.
     *
     * \param gpu_postprocess_frame GPU buffer for post-processed frame data.
     * \param gpu_convolution_buffer GPU buffer for convolution data.
     */
    void insert_convolution(float* gpu_postprocess_frame, float* gpu_convolution_buffer);

    /*! \brief Inserts the renormalization function into the function vector.
     *
     * \param gpu_postprocess_frame GPU buffer for post-processed frame data.
     */
    void insert_renormalize(float* gpu_postprocess_frame);

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
            spdlog::trace("[Postprocessing] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            spdlog::trace("[Postprocessing] [update_setting] {}", typeid(T).name());
            onrestart_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief Performs convolution on composite data.
     *
     * \param gpu_postprocess_frame GPU buffer for post-processed frame data.
     * \param gpu_convolution_buffer GPU buffer for convolution data.
     * \param divide_convolution_enabled Indicates if divide convolution is enabled.
     */
    void convolution_composite(float* gpu_postprocess_frame, float* gpu_convolution_buffer, bool divide_convolution_enabled);

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

    /*! \brief GPU buffer for the convolution kernel. */
    cuda_tools::CudaUniquePtr<cuComplex> gpu_kernel_buffer_;
    /*! \brief GPU buffer for complex data. */
    cuda_tools::CudaUniquePtr<cuComplex> cuComplex_buffer_;
    /*! \brief GPU buffer for HSV data. */
    cuda_tools::CudaUniquePtr<float> hsv_arr_;

    /*! \brief Result of the reduce operation used for renormalization. */
    cuda_tools::CudaUniquePtr<double> reduce_result_;

    /*! \brief Vector of functions for compute operations. */
    FunctionVector& fn_compute_vect_;

    /*! \brief Core buffer environment. */
    CoreBuffersEnv& buffers_;

    /*! \brief Descriptor for input frame size. */
    const camera::FrameDescriptor& fd_;

    /*! \brief CUFFT handle for the convolution plan. */
    CufftHandle convolution_plan_;

    /*! \brief CUDA stream for asynchronous operations. */
    const cudaStream_t& stream_;

    /*! \brief Container for real-time settings. */
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
    /*! \brief Container for settings applied on restart. */
    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
/*! \brief Checks if a setting exists in the Postprocessing class.
 *
 * \tparam T Type of the setting.
 */
template <typename T>
struct has_setting<T, compute::Postprocessing> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes