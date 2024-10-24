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

#define REALTIME_SETTINGS                          \
    holovibes::settings::ArteryMaskEnabled,        \
    holovibes::settings::OtsuEnabled,              \
    holovibes::settings::ConvolutionMatrix,        \
    holovibes::settings::ImageType,                \
    holovibes::settings::TimeWindow,               \
    holovibes::settings::VesselnessSigma           \


#define ALL_SETTINGS REALTIME_SETTINGS

// clang-format on

using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
struct CoreBuffersEnv;
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
             CoreBuffersEnv& buffers,
             const camera::FrameDescriptor& input_fd,
             const cudaStream_t& stream,
             InitSettings settings)
        : gpu_kernel_buffer_()
        , cuComplex_buffer_()
        , fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , fd_(input_fd)
        , convolution_plan_(input_fd.height, input_fd.width, CUFFT_C2C)
        , stream_(stream)
        , realtime_settings_(settings)
    {
    }

    /*! \brief Initialize convolution by allocating the corresponding buffer */
    void init();

    /*! \brief Free the ressources for the postprocessing */
    void dispose();

    /*! \brief Insert artery mask computing */
    void insert_show_artery();

    template <typename T>
    inline void update_setting(T setting)
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            LOG_TRACE("[Analysis] [update_setting] {}", typeid(T).name());
            realtime_settings_.update_setting(setting);
        }
    }

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
    }

    /*! \brief Buffer used for convolution */
    cuda_tools::CudaUniquePtr<cuComplex> gpu_kernel_buffer_;

    /* \brief Gaussian kernel used for flat field correction */
    std::vector<float> gaussian_kernel_;

    /*! \brief TODO comment */
    cuda_tools::CudaUniquePtr<cuComplex> cuComplex_buffer_;

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

    /*! \brief Get the number of image for the mean mask*/
    int number_image_mean_ = 0;

    /*! \brief Get the mean image*/
    float* m0_ff_sum_image_;

    /*! \brief Buffer of size 'batch_moment' TODO refaire ca to compute the mean of m0 imgs */
    float* buffer_m0_ff_img_ = nullptr;

    int time_window_;
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::Analysis> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
