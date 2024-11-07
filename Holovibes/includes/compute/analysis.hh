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
#include "cublas_handle.hh"
#include "logger.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::ArteryMaskEnabled,        \
    holovibes::settings::VeinMaskEnabled,          \
    holovibes::settings::OtsuEnabled,              \
    holovibes::settings::OtsuKind,                 \
    holovibes::settings::OtsuWindowSize,           \
    holovibes::settings::OtsuLocalThreshold,       \
    holovibes::settings::BwareafiltEnabled,        \
    holovibes::settings::BwareafiltN,        \
    holovibes::settings::ConvolutionMatrix,        \
    holovibes::settings::ImageType,                \
    holovibes::settings::TimeWindow,               \
    holovibes::settings::VesselnessSigma,          \
    holovibes::settings::MinMaskArea


#define ALL_SETTINGS REALTIME_SETTINGS

// clang-format on

using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
struct CoreBuffersEnv;
struct VesselnessMaskEnv;
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
             VesselnessMaskEnv& vesselness_mask_env,
             const cudaStream_t& stream,
             InitSettings settings)
        : gaussian_kernel_buffer_()
        , cuComplex_buffer_()
        , fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , fd_(input_fd)
        , vesselness_mask_env_(vesselness_mask_env)
        , convolution_plan_(input_fd.height, input_fd.width, CUFFT_C2C)
        , stream_(stream)
        , realtime_settings_(settings)
    {
        [[maybe_unused]] auto status = cublasCreate_v2(&cublas_handler_);
        cublasSetStream(cublas_handler_, stream_);
    }

    /*! \brief Initialize convolution by allocating the corresponding buffer */
    void init();

    /*! \brief Free the ressources for the postprocessing */
    void dispose();

    /*! \brief Insert artery mask computing */
    void insert_show_artery();

    /*! \brief TODO */
    void insert_otsu();

    /*! \brief TODO */
    void insert_bwareafilt();

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

    /*! \brief Buffer used for gaussian blur convolution kernel */
    cuda_tools::CudaUniquePtr<cuComplex> gaussian_kernel_buffer_;

    /*! \brief TODO comment */
    cuda_tools::CudaUniquePtr<cuComplex> cuComplex_buffer_;

    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;

    /*! \brief Main buffers */
    CoreBuffersEnv& buffers_;

    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;

    /*! \brief Vesselness mask environment */
    VesselnessMaskEnv& vesselness_mask_env_;

    /*! \brief Plan used for the convolution (frame width, frame height, cufft_c2c) */
    CufftHandle convolution_plan_;

    /*! \brief Cublas handler used for matrices multiplications */
    cublasHandle_t cublas_handler_;

    /*! \brief Compute stream to perform pipe computation */
    const cudaStream_t& stream_;

    // To delete
    cuda_tools::CudaUniquePtr<float> data_csv_;

    // To delete
    cuda_tools::CudaUniquePtr<float> data_csv_avg_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;

    /*! \brief TODO comment */
    cuda_tools::CudaUniquePtr<size_t> size_t_buffer_1_;
    /*! \brief TODO comment */
    cuda_tools::CudaUniquePtr<size_t> size_t_buffer_2_;
    /*! \brief TODO comment */
    cuda_tools::CudaUniquePtr<size_t> size_t_buffer_3_;
    cuda_tools::CudaUniquePtr<size_t> size_t_gpu_;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::Analysis> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
