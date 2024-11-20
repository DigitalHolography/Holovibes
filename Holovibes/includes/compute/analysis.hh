/*! \file analysis.hh
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
#include "tools_analysis.cuh"

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
    holovibes::settings::BwareaopenEnabled,        \
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
 * \brief Class containing pipe methods for moments analysis
 */
class Analysis
{
  public:
    /*! \brief Constructor */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Analysis(std::shared_ptr<FunctionVector> fn_compute_vect,
             CoreBuffersEnv& buffers,
             const camera::FrameDescriptor& input_fd,
             VesselnessMaskEnv& vesselness_mask_env,
             const cudaStream_t& stream,
             InitSettings settings)
        : gaussian_128_kernel_buffer_()
        , cuComplex_buffer_()
        , fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , fd_(input_fd)
        , vesselness_mask_env_(vesselness_mask_env)
        , convolution_plan_(input_fd.height, input_fd.width, CUFFT_C2C)
        , stream_(stream)
        , realtime_settings_(settings)
    {
        // Create for Analysis its own cublas handler associated to its personal cuda stream
        [[maybe_unused]] auto status = cublasCreate_v2(&cublas_handler_);
        cublasSetStream(cublas_handler_, stream_);

        // TODO: remove this when done
        // Load valid moment test data for debugging purpose
        const size_t frame_res = fd_.get_frame_res();

        float* data_csv_cpu = load_CSV_to_float_array("C:/Users/Karachayevsk/Documents/Holovibes/data_n.csv");
        m0_ff_img_csv_.resize(frame_res);
        cudaXMemcpy(m0_ff_img_csv_, data_csv_cpu, frame_res * sizeof(float), cudaMemcpyHostToDevice);
        cudaXStreamSynchronize(stream_);
        delete[] data_csv_cpu;

        data_csv_cpu = load_CSV_to_float_array("C:/Users/Karachayevsk/Documents/Holovibes/f_AVG_mean.csv");
        f_avg_csv_.resize(frame_res);
        cudaXMemcpy(f_avg_csv_, data_csv_cpu, frame_res * sizeof(float), cudaMemcpyHostToDevice);
        cudaXStreamSynchronize(stream_);
        delete[] data_csv_cpu;

        data_csv_cpu = load_CSV_to_float_array("C:/Users/Karachayevsk/Documents/Holovibes/vascularPulse.csv");
        vascular_pulse_csv_.resize(506);
        cudaXMemcpy(vascular_pulse_csv_, data_csv_cpu, 506 * sizeof(float), cudaMemcpyHostToDevice);
        cudaXStreamSynchronize(stream_);
        delete[] data_csv_cpu;

        data_csv_cpu = load_CSV_to_float_array("C:/Users/Karachayevsk/Documents/Holovibes/R_VascularPulse.csv");
        R_VascularPulse_csv_.resize(frame_res);
        cudaXMemcpy(R_VascularPulse_csv_, data_csv_cpu, frame_res * sizeof(float), cudaMemcpyHostToDevice);
        cudaXStreamSynchronize(stream_);
        delete[] data_csv_cpu;
    }

    /*! \brief Initialize convolution by allocating the corresponding buffer */
    void init();

    /*! \brief Free the ressources for the analysis */
    void dispose();

    /*! \brief Insert artery mask computing */
    void insert_show_artery();

    /*! \brief Insert barycentres*/
    void insert_barycentres();

    /*! \brief Insert otsu computation (binarisation) */
    void insert_otsu();

    /*! \brief Insert bw area filter compution (keep the biggest connected component from binarised image) */
    void insert_bwareafilt();

    /*! \brief Insert bw area open compution (keep connected component bigger than a parameter from binarised image) */
    void insert_bwareaopen();

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

    /*! \brief Buffer used for gaussian blur 128x128 convolution kernel */
    cuda_tools::CudaUniquePtr<cuComplex> gaussian_128_kernel_buffer_;

    /*! \brief TODO comment */
    cuda_tools::CudaUniquePtr<cuComplex> cuComplex_buffer_;

    /*! \brief Vector function in which we insert the processing */
    std::shared_ptr<FunctionVector> fn_compute_vect_;

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
    cuda_tools::CudaUniquePtr<float> m0_ff_img_csv_;

    // To delete
    cuda_tools::CudaUniquePtr<float> f_avg_csv_;

    // To delete
    cuda_tools::CudaUniquePtr<float> R_VascularPulse_csv_;

    // To delete
    cuda_tools::CudaUniquePtr<float> vascular_pulse_csv_;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;

    cuda_tools::CudaUniquePtr<uint> uint_buffer_1_;
    cuda_tools::CudaUniquePtr<uint> uint_buffer_2_;
    cuda_tools::CudaUniquePtr<float> float_buffer_;
    cuda_tools::CudaUniquePtr<uint> otsu_histo_buffer_;
};
} // namespace holovibes::compute

namespace holovibes
{
template <typename T>
struct has_setting<T, compute::Analysis> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
