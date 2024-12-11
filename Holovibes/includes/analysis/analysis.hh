/*! \file analysis.hh
 *
 * \brief Implementation of postprocessing features on complex buffers.
 */
#pragma once

#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "cufft_handle.hh"
#include "cublas_handle.hh"
#include "tools_analysis_debug.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::ArteryMaskEnabled,        \
    holovibes::settings::VeinMaskEnabled,          \
    holovibes::settings::ChartMeanVesselsEnabled,  \
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

#pragma endregion

// clang-format on

using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
struct CoreBuffersEnv;
struct VesselnessMaskEnv;
struct MomentsEnv;
struct FirstMaskChoroidEnv;
struct OtsuEnv;
struct BwAreaEnv;
} // namespace holovibes

namespace holovibes::analysis
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
             MomentsEnv& moments_env,
             ChartMeanVesselsEnv& chart_mean_vessels_env,
             const cudaStream_t& stream,
             InitSettings settings)
        : cuComplex_buffer_()
        , fn_compute_vect_(fn_compute_vect)
        , buffers_(buffers)
        , fd_(input_fd)
        , vesselness_mask_env_(vesselness_mask_env)
        , moments_env_(moments_env)
        , stream_(stream)
        , realtime_settings_(settings)
        , chart_mean_vessels_env_(chart_mean_vessels_env)
    {
        // Create for Analysis its own cublas handler associated to its personal cuda stream
        cublasCreate_v2(&cublas_handler_);
        cublasSetStream(cublas_handler_, stream_);

        // TODO: remove everything below when done
        // Load valid moment test data for debugging purpose
        const size_t frame_res = fd_.get_frame_res();

        m0_bin_video_.resize(512 * 512 * 506);
        load_bin_video_file(RELATIVE_PATH("../../Obj_M0_data_video_permuted.bin"), m0_bin_video_, stream_);

        m1_bin_video_.resize(512 * 512 * 506);
        load_bin_video_file(RELATIVE_PATH("../../Obj_M1_data_video_permuted.bin"), m1_bin_video_, stream_);
    }

    /*!
     * \brief Initializes GPU buffers, kernel parameters, and resources required for vesselness
     *        filtering and vascular pulse computation.
     *
     * This function prepares various structures, allocates GPU memory, computes Gaussian
     * kernel parameters, and initializes vesselness filter buffers and other resources.
     *
     * Key operations include:
     * - Calculating Gaussian kernel dimensions and parameters.
     * - Allocating and resizing buffers for vesselness and vascular processing.
     * - Initializing Gaussian derivative kernels for vesselness computations.
     * - Preparing circular video buffers for temporal data.
     */
    void init();

    /*! \brief Insert mask computing */
    void insert_first_analysis_masks();

    /*! \brief Insert artery mask*/
    void insert_artery_mask();

    /*! \brief Insert vein mask*/
    void insert_vein_mask();

    /*! \brief Insert choroid mask */
    void insert_choroid_mask();

    /*! \brief Insert both masks*/
    void insert_vesselness();

    /*! \brief Insert chart compute*/
    void insert_chart();

    /*! \brief Getter for the mask result buffer */
    float* get_mask_result();

    /*! \brief Getter for the mask number of non zero count */
    size_t get_mask_nnz();

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
    /*!
     * \brief Initializes parameters for a vesselness filter by computing and transposing
     *        derivative Gaussian kernels on the GPU.
     *
     * This function generates normalized coordinate lists for x and y dimensions, computes
     * the X and Y derivative Gaussian kernels, multiplies them to form a filter kernel,
     * and transposes the result. All computations are performed on the GPU.
     *
     * \param [out] result_transpose Pointer to the GPU memory where the transposed result is stored.
     * \param [out] target Pointer to the GPU memory where the intermediate result is stored.
     * \param [in] sigma Standard deviation of the Gaussian kernel.
     * \param [in] x_size Size of the x-dimension.
     * \param [in] y_size Size of the y-dimension.
     * \param [in] x_lim Range limit for normalization in the x-dimension.
     * \param [in] y_lim Range limit for normalization in the y-dimension.
     * \param [in] p Order of the derivative in the x-dimension.
     * \param [in] q Order of the derivative in the y-dimension.
     * \param [in] stream CUDA stream for asynchronous execution.
     */
    void init_params_vesselness_filter(float* result_transpose,
                                       float* target,
                                       float sigma,
                                       int x_size,
                                       int y_size,
                                       int x_lim,
                                       int y_lim,
                                       int p,
                                       int q,
                                       cudaStream_t stream);

    /*! \brief To be remove, only for test */
    void insert_bin_moments();

    /*! \brief Compute pretreatment to be able to do the different masks, such as the temporal mean of M0 */
    void compute_pretreatment();

    /*! \brief Compute vesselness response, which includes the first vesselness mask */
    void compute_vesselness_response();

    /*! \brief Compute the barycentres and the circle mask, which includes the mask vesselness clean */
    void compute_barycentres_and_circle_mask();

    /*! \brief Compute the first correlation, which leads to R_vascular_pulse */
    void compute_correlation();

    /*! \brief Compute the segment vessels, which includes quantizedVesselCorrelation_, an image with only values from 1
     * to 5 */
    void compute_segment_vessels();

    /*!
     * \brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
            return realtime_settings_.get<T>().value;
    }

    /*! \brief Temporary complex buffer used for FFT computations */
    cuda_tools::CudaUniquePtr<cuComplex> cuComplex_buffer_;

    /*! \brief Vector function in which we insert the processing */
    std::shared_ptr<FunctionVector> fn_compute_vect_;

    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;

    /*! \brief Main buffers */
    CoreBuffersEnv& buffers_;

    /*! \brief Vesselness mask environment */
    VesselnessMaskEnv& vesselness_mask_env_;

    /*! \brief Vesselness filter buffers struct */
    VesselnessFilterEnv vesselness_filter_struct_;

    /*! \brief FirstMaskChoroidEnv buffers struct */
    FirstMaskChoroidEnv first_mask_choroid_struct_;

    /*! \brief Reference to the OtsuEnv to get access to otsu buffers */
    OtsuEnv otsu_env_;

    /*! \brief Reference to the BwAreaEnv to get access to otsu buffers */
    BwAreaEnv bw_area_env_;

    /*! \brief Reference to the MomentsEnv to get access to moments buffers */
    MomentsEnv& moments_env_;

    /*! \brief Reference to the ChartMeanVesselsEnv to get access to chart display queue */
    ChartMeanVesselsEnv& chart_mean_vessels_env_;

    /*! \brief Cublas handler used for matrices multiplications */
    cublasHandle_t cublas_handler_;

    /*! \brief Compute stream to perform pipe computation */
    const cudaStream_t& stream_;

    // To delete
    cuda_tools::CudaUniquePtr<float> m0_bin_video_;

    // To delete
    cuda_tools::CudaUniquePtr<float> m1_bin_video_;

    // To delete
    size_t i_ = 0;

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;

    /*! \brief Buffer containing the final resulting mask */
    cuda_tools::CudaUniquePtr<float> mask_result_buffer_;
};
} // namespace holovibes::analysis

namespace holovibes
{
template <typename T>
struct has_setting<T, analysis::Analysis> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
