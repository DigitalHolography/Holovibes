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
#include "logger.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                               \
    holovibes::settings::ImageType,                     \
    holovibes::settings::RenormEnabled,                 \
    holovibes::settings::ConvolutionMatrix,             \
    settings::ConvolutionEnabled,                       \
    settings::DivideConvolutionEnabled


#define ONRESTART_SETTINGS                          \
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
 * \brief #TODO Add a description for this class
 */
class Postprocessing
{
  public:
    /*! \brief Constructor */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Postprocessing(FunctionVector& fn_compute_vect,
                   CoreBuffersEnv& buffers,
                   const camera::FrameDescriptor& input_fd,
                   const cudaStream_t& stream,
                   InitSettings settings)
        : gpu_kernel_buffer_()
        , cuComplex_buffer_()
        , hsv_arr_()
        , reduce_result_(1) // allocate an unique double
        , fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , fd_(input_fd)
        , convolution_plan_(input_fd.height, input_fd.width, CUFFT_C2C)
        , stream_(stream)
        , realtime_settings_(settings)
        , onrestart_settings_(settings)
    {
    }

    /*! \brief Initialize convolution by allocating the corresponding buffer */
    void init();

    /*! \brief Free the ressources for the postprocessing */
    void dispose();

    /*! \brief Insert the Convolution function. */
    void insert_convolution(float* gpu_postprocess_frame, float* gpu_convolution_buffer);

    /*! \brief Insert the normalization function. */
    void insert_renormalize(float* gpu_postprocess_frame);

    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            LOG_TRACE("[PostProcessing] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            LOG_TRACE("[PostProcessing] [update_setting] {}", typeid(T).name());
            onrestart_settings_.update_setting(setting);
        }
    }

  private:
    /*! \brief Used only when the image is composite convolution to do a convolution on each component */
    void
    convolution_composite(float* gpu_postprocess_frame, float* gpu_convolution_buffer, bool divide_convolution_enabled);

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

    cuda_tools::CudaUniquePtr<cuComplex> gpu_kernel_buffer_;
    cuda_tools::CudaUniquePtr<cuComplex> cuComplex_buffer_;
    cuda_tools::CudaUniquePtr<float> hsv_arr_;

    /*! \brief Result of the reduce operation of the current frame used to renormalize the frames */
    cuda_tools::CudaUniquePtr<double> reduce_result_;

    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;

    /*! \brief Main buffers */
    CoreBuffersEnv& buffers_;

    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;

    /*! \brief Plan used for the convolution (frame width, frame height, cufft_c2c) */
    CufftHandle convolution_plan_;

    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::Postprocessing> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
