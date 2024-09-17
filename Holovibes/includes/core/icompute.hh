/*! \file
 *
 * \brief Stores functions helping the editing of the images.
 */
#pragma once

#include <atomic>
#include <memory>

#include "queue.hh"

#include "rect.hh"
#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "cufft_handle.hh"
#include "chart_point.hh"
#include "concurrent_deque.hh"
#include "enum_window_kind.hh"
#include "enum_record_mode.hh"
#include "global_state_holder.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#define ICS holovibes::ICompute::Setting

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                                        \
    holovibes::settings::ImageType,                              \
    holovibes::settings::X,                                      \
    holovibes::settings::Y,                                      \
    holovibes::settings::P,                                      \
    holovibes::settings::Q,                                      \
    holovibes::settings::Filter2d,                               \
    holovibes::settings::CurrentWindow,                          \
    holovibes::settings::LensViewEnabled,                        \
    holovibes::settings::ChartDisplayEnabled,                    \
    holovibes::settings::Filter2dEnabled,                        \
    holovibes::settings::Filter2dViewEnabled,                    \
    holovibes::settings::FftShiftEnabled,                        \
    holovibes::settings::RawViewEnabled,                         \
    holovibes::settings::CutsViewEnabled,                        \
    holovibes::settings::RenormEnabled,                          \
    holovibes::settings::ReticleScale,                           \
    holovibes::settings::Filter2dN1,                             \
    holovibes::settings::Filter2dN2,                             \
    holovibes::settings::Filter2dSmoothLow,                      \
    holovibes::settings::Filter2dSmoothHigh,                     \
    holovibes::settings::TimeTransformationSize,                 \
    holovibes::settings::TimeTransformation,                     \
    holovibes::settings::TimeTransformationCutsOutputBufferSize, \
    holovibes::settings::CompositeKind,                          \
    holovibes::settings::CompositeAutoWeights,                   \
    holovibes::settings::RGB,                                    \
    holovibes::settings::HSV

#define PIPEREFRESH_SETTINGS                                     \
    holovibes::settings::XY,                                     \
    holovibes::settings::XZ,                                     \
    holovibes::settings::YZ,                                     \
    holovibes::settings::BatchSize

#define ALL_SETTINGS REALTIME_SETTINGS, PIPEREFRESH_SETTINGS

// clang-format on
#pragma endregion

namespace holovibes
{
/*! \struct CoreBuffersEnv
 *
 * \brief Struct containing main buffers used by the pipe.
 */
struct CoreBuffersEnv
{
    /*! \brief Input buffer. Contains only one frame. We fill it with the input frame */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_spatial_transformation_buffer = nullptr;

    /*! \brief Float buffer. Contains only one frame.
     *
     * We fill it with the correct computed p frame converted to float.
     */
    cuda_tools::CudaUniquePtr<float> gpu_postprocess_frame = nullptr;

    /*! \brief Size in components (size in byte / sizeof(float)) of the gpu_postprocess_frame.
     *
     * Could be removed by changing gpu_postprocess_frame type to cuda_tools::Array.
     */
    unsigned int gpu_postprocess_frame_size = 0;

    /*! \brief Float XZ buffer of 1 frame, filled with the correct computed p XZ frame. */
    cuda_tools::CudaUniquePtr<float> gpu_postprocess_frame_xz = nullptr;

    /*! \brief Float YZ buffer of 1 frame, filled with the correct computed p YZ frame. */
    cuda_tools::CudaUniquePtr<float> gpu_postprocess_frame_yz = nullptr;

    /*! \brief Unsigned Short output buffer of 1 frame, inserted after all postprocessing on float_buffer */
    cuda_tools::CudaUniquePtr<unsigned short> gpu_output_frame = nullptr;

    /*! \brief Unsigned Short XZ output buffer of 1 frame, inserted after all postprocessing on float_buffer_cut_xz */
    cuda_tools::CudaUniquePtr<unsigned short> gpu_output_frame_xz = nullptr;

    /*! \brief Unsigned Short YZ output buffer of 1 frame, inserted after all postprocessing on float_buffer_cut_yz */
    cuda_tools::CudaUniquePtr<unsigned short> gpu_output_frame_yz = nullptr;

    /*! \brief Contains only one frame used only for convolution */
    cuda_tools::CudaUniquePtr<float> gpu_convolution_buffer = nullptr;

    /*! \brief Complex filter2d frame used to store the output_frame */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_complex_filter2d_frame = nullptr;

    /*! \brief Float Filter2d frame used to store the gpu_complex_filter2d_frame */
    cuda_tools::CudaUniquePtr<float> gpu_float_filter2d_frame = nullptr;

    /*! \brief Filter2d frame used to store the gpu_float_filter2d_frame */
    cuda_tools::CudaUniquePtr<unsigned short> gpu_filter2d_frame = nullptr;

    /*! \brief Filter2d mask applied to gpu_spatial_transformation_buffer */
    cuda_tools::CudaUniquePtr<float> gpu_filter2d_mask = nullptr;

    /*! \brief InputFilter mask */
    cuda_tools::CudaUniquePtr<float> gpu_input_filter_mask = nullptr;
};

/*! \struct BatchEnv
 *
 * \brief Struct containing variables related to the batch in the pipe
 */
struct BatchEnv
{
    /*! \brief Current frames processed in the batch
     *
     * At index 0, batch_size frames are enqueued, spatial transformation is
     * also executed in batch
     * Batch size frames are enqueued in the gpu_time_transformation_queue
     * This is done for perfomances reasons
     *
     * The variable is incremented by batch_size until it reaches timestride in
     * enqueue_multiple, then it is set back to 0
     */
    uint batch_index = 0;
};

/*! \struct TimeTransformationEnv
 *
 * \brief Struct containing variables related to STFT shared by multiple
 * features of the pipe.
 */
struct TimeTransformationEnv
{
    /*! \brief STFT Queue. It accumulates input frames after spatial FFT.
     *
     * Contains time_transformation_size frames.
     * Frames are accumulated in order to apply STFT only when
     * the frame counter is equal to time_stride.
     */
    std::unique_ptr<Queue> gpu_time_transformation_queue = nullptr;

    /*! \brief STFT buffer. Contains the result of the STFT done on the STFT queue.
     *
     * Contains time_transformation_size frames.
     */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_p_acc_buffer = nullptr;

    /*! \brief STFT XZ Queue. Contains the ouput of the STFT on slice XZ.
     *
     * Enqueued with gpu_float_buffer or gpu_ushort_buffer.
     */
    std::unique_ptr<Queue> gpu_output_queue_xz = nullptr;

    /*! \brief STFT YZ Queue. Contains the ouput of the STFT on slice YZ.
     *
     * Enqueued with gpu_float_buffer or gpu_ushort_buffer.
     */
    std::unique_ptr<Queue> gpu_output_queue_yz = nullptr;

    /*! \brief Plan 1D used for the STFT. */
    cuda_tools::CufftHandle stft_plan;

    /*! \brief Hold the P frame after the time transformation computation. */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_p_frame;

    /*! \name PCA time transformation
     * \{
     */
    cuda_tools::CudaUniquePtr<cuComplex> pca_cov = nullptr;
    cuda_tools::CudaUniquePtr<float> pca_eigen_values = nullptr;
    cuda_tools::CudaUniquePtr<int> pca_dev_info = nullptr;
    /*! \} */
};

/*! \struct ChartEnv
 *
 * \brief Structure containing variables related to the chart display and
 * recording.
 */
struct ChartEnv
{
    std::unique_ptr<ConcurrentDeque<ChartPoint>> chart_display_queue_ = nullptr;
    std::unique_ptr<ConcurrentDeque<ChartPoint>> chart_record_queue_ = nullptr;
    unsigned int nb_chart_points_to_record_ = 0;
};

/*! \struct ImageAccEnv
 *
 * \brief #TODO Add a description for this struct
 */
struct ImageAccEnv
{
    /*! \brief Frame to temporaly store the average on XY view */
    cuda_tools::CudaUniquePtr<float> gpu_float_average_xy_frame = nullptr;

    /*! \brief Queue accumulating the XY computed frames. */
    std::unique_ptr<Queue> gpu_accumulation_xy_queue = nullptr;

    /*! \brief Frame to temporaly store the average on XZ view */
    cuda_tools::CudaUniquePtr<float> gpu_float_average_xz_frame = nullptr;

    /*! \brief Queue accumulating the XZ computed frames. */
    std::unique_ptr<Queue> gpu_accumulation_xz_queue = nullptr;

    /*! \brief Frame to temporaly store the average on YZ axis */
    cuda_tools::CudaUniquePtr<float> gpu_float_average_yz_frame = nullptr;

    /*! \brief Queue accumulating the YZ computed frames. */
    std::unique_ptr<Queue> gpu_accumulation_yz_queue = nullptr;
};

/*! \class ICompute
 *
 * \brief Stores functions helping the editing of the images.
 *
 * Stores all the functions that will be used before doing any sort
 * of editing to the image (i.e. refresh functions or caller).
 */
class ICompute
{
  public:
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    ICompute(BatchInputQueue& input, Queue& output, Queue& record, const cudaStream_t& stream, InitSettings settings)
        : input_queue_(input)
        , gpu_output_queue_(output)
        , record_queue_(record)
        , stream_(stream)
        , realtime_settings_(settings)
        , pipe_refresh_settings_(settings)
    {
        // Initialize the array of settings to false except for the refresh
        for (auto& setting : settings_requests_)
            setting.store(false, std::memory_order_relaxed);
        settings_requests_[static_cast<int>(ICS::RefreshEnabled)] = true;

        camera::FrameDescriptor fd = input_queue_.get_fd();
        int inembed[1] = {static_cast<int>(setting<settings::TimeTransformationSize>())};
        int zone_size = static_cast<int>(fd.get_frame_res());

        plan_unwrap_2d_.plan(fd.width, fd.height, CUFFT_C2C);

        update_spatial_transformation_parameters();

        time_transformation_env_.stft_plan
            .planMany(1, inembed, inembed, zone_size, 1, inembed, zone_size, 1, CUFFT_C2C, zone_size);

        fd.depth = 8;
        // FIXME-CAMERA : WTF depth 8 ==> maybe a magic value for complex mode
        time_transformation_env_.gpu_time_transformation_queue.reset(
            new Queue(fd, setting<settings::TimeTransformationSize>()));

        if (setting<settings::ImageType>() == ImgType::Composite)
        {
            // Grey to RGB
            zone_size *= 3;
            buffers_.gpu_postprocess_frame_size *= 3;
        }

        buffers_.gpu_postprocess_frame_size = zone_size;

        // Allocate the buffers
        int err = !buffers_.gpu_output_frame.resize(zone_size);
        err += !buffers_.gpu_postprocess_frame.resize(buffers_.gpu_postprocess_frame_size);
        err += !time_transformation_env_.gpu_p_frame.resize(buffers_.gpu_postprocess_frame_size);
        err += !buffers_.gpu_complex_filter2d_frame.resize(buffers_.gpu_postprocess_frame_size);
        err += !buffers_.gpu_float_filter2d_frame.resize(buffers_.gpu_postprocess_frame_size);
        err += !buffers_.gpu_filter2d_frame.resize(buffers_.gpu_postprocess_frame_size);
        err += !buffers_.gpu_filter2d_mask.resize(zone_size);
        err += !buffers_.gpu_input_filter_mask.resize(zone_size);

        if (err != 0)
            throw std::exception(cudaGetErrorString(cudaGetLastError()));
    }

    template <typename T>
    inline void update_setting_icompute(T setting)
    {
        spdlog::trace("[ICompute] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting_v<T, decltype(realtime_settings_)>)
            realtime_settings_.update_setting(setting);
        else if constexpr (has_setting_v<T, decltype(pipe_refresh_settings_)>)
            pipe_refresh_settings_.update_setting(setting);
    }

    inline void icompute_pipe_refresh_apply_updates() { pipe_refresh_settings_.apply_updates(); }

    /*! \brief Execute one iteration of the ICompute.
     *
     * Checks the number of frames in input queue that must at least time_transformation_size.
     * Call each function of the ICompute.
     * Enqueue the output frame contained in gpu_output_buffer.
     * Dequeue one frame of the input queue.
     * Check if a ICompute refresh has been requested.
     *
     * The ICompute can not be interrupted for parameters changes until the refresh method is called.
     */
    virtual void exec() = 0;

    /*! \brief enum class for the settings that can be requested: settings that change the pipeline. */
    enum class Setting
    {
        Unwrap2D = 0,

        // These 4 autocontrast settings are set to false by & in renderer.cc
        // it's not clean
        Autocontrast,
        AutocontrastSliceXZ,
        AutocontrastSliceYZ,
        AutocontrastFilter2D,

        Refresh,
        RefreshEnabled,
        UpdateTimeTransformationSize,
        ChartDisplay,
        DisableChartDisplay,
        DisableChartRecord,
        RawView,
        DisableRawView,
        Filter2DView,
        DisableFilter2DView,
        Termination,
        TimeTransformationCuts,
        DeleteTimeTransformationCuts,
        UpdateBatchSize,
        UpdateTimeStride,
        DisableLensView,
        FrameRecord,
        DisableFrameRecord,
        ClearImgAccu,
        Convolution,
        DisableConvolution,
        Filter,
        DisableFilter,

        // Add other setting here

        Count // Used to create the array
    };

    /*! \name Request Settings
     * \{
     */
    /*! \brief Whether the setting is requested.
     *  \tparam T The type of the setting.
     *  \return The value of the setting.
     */
    std::atomic<bool>& is_requested(Setting setting);

    /*! \brief Request the setting (like request a filter2D in the pipeline) and call @ref
     * holovibes::ICompute::request_refresh "request_refresh".
     * \param setting The setting to be requested.
     */
    void request(Setting setting);

    /*! \brief Set the setting to the value and but do not call @ref holovibes::ICompute::request_refresh
     * "request_refresh".
     * \param setting The setting to be set.
     * \param value The value to be set.
     */
    void set_requested(Setting setting, bool value);

    /*! \brief Clear the request of the setting.
     * \param setting The setting to be cleared.
     */
    void clear_request(Setting setting);

    std::optional<unsigned int> get_chart_record_requested() const { return chart_record_requested_; }

    void request_refresh();

    void request_autocontrast(WindowKind kind);

    void request_record_chart(unsigned int nb_chart_points_to_record);
    /*! \} */

    /*! \name Queue getters
     * \{
     */
    std::unique_ptr<Queue>& get_stft_slice_queue(int i);

    virtual std::unique_ptr<Queue>& get_lens_queue() = 0;

    virtual std::unique_ptr<Queue>& get_raw_view_queue();

    virtual std::unique_ptr<Queue>& get_filter2d_view_queue();

    virtual std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_display_queue();

    virtual std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_record_queue();
    /*! \} */

  protected:
    virtual void refresh() = 0;

    virtual bool update_time_transformation_size(const unsigned short time_transformation_size);

    /*! \name Resources management
     * \{
     */
    virtual void update_spatial_transformation_parameters();

    void init_cuts();

    void dispose_cuts();
    /*! \} */

    /*! \name ICCompute operators
     * \{
     */
    ICompute& operator=(const ICompute&) = delete;

    ICompute(const ICompute&) = delete;

    virtual ~ICompute() {}
    /*! \} */

  protected:
    /*! \brief Counting pipe iteration, in order to update fps only every 100 iterations. */
    unsigned int frame_count_{0};

    /*! \name Queues */
    /*! \brief Reference on the input queue */
    BatchInputQueue& input_queue_;

    /*! \brief Reference on the output queue */
    Queue& gpu_output_queue_;

    /*! \brief Reference on the record queue */
    Queue& record_queue_;

    /*! \brief Queue storing raw frames used by raw view */
    std::unique_ptr<Queue> gpu_raw_view_queue_{nullptr};

    /*! \brief Queue storing filter2d frames */
    std::unique_ptr<Queue> gpu_filter2d_view_queue_{nullptr};

    /*! \name Compute environment */
    /*! \brief Main buffers. */
    CoreBuffersEnv buffers_;

    /*! \brief Batch environment */
    BatchEnv batch_env_;

    /*! \brief STFT environment. */
    TimeTransformationEnv time_transformation_env_;

    /*! \brief Chart environment. */
    ChartEnv chart_env_;

    /*! \brief Image accumulation environment */
    ImageAccEnv image_acc_env_;

    /*! \name Cuda */
    /*! \brief Pland 2D. Used for spatial fft performed on the complex input frame. */
    cuda_tools::CufftHandle spatial_transformation_plan_;

    /*! \brief Pland 2D. Used for unwrap 2D. */
    cuda_tools::CufftHandle plan_unwrap_2d_;

    /*! \brief Compute stream to perform pipe computation */
    const cudaStream_t& stream_;

    /*! \name Requested settings */
    /*! \brief Requested chart record. */
    std::atomic<std::optional<unsigned int>> chart_record_requested_{std::nullopt};

    /*! \brief Array of atomic bools to store the requested settings. */
    std::array<std::atomic<bool>, static_cast<int>(Setting::Count)> settings_requests_{};

    /*! \name Settings containers
     * \{
     */
    /*! \brief Container for the realtime settings. */
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;

    /*! \brief Container for the pipe refresh settings. */
    DelayedSettingsContainer<PIPEREFRESH_SETTINGS> pipe_refresh_settings_;
    /*! \} */

  private:
    /**
     * @brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting_v<T, decltype(realtime_settings_)>)
            return realtime_settings_.get<T>().value;

        if constexpr (has_setting_v<T, decltype(pipe_refresh_settings_)>)
            return pipe_refresh_settings_.get<T>().value;
    }

    /*! \brief Performs tasks specific to the current time transformation setting.
     *  \param size The size for time transformation.
     */
    void perform_time_transformation_setting_specific_tasks(const unsigned short size);

    /*! \brief Updates the STFT configuration based on the time transformation size.
     *  \param size The size for time transformation.
     */
    void update_stft(const unsigned short size);

    /*! \brief Updates the PCA configuration based on the time transformation size.
     *  \param size The size for time transformation.
     */
    void update_pca(const unsigned short size);
};
} // namespace holovibes

namespace holovibes
{
template <typename T>
struct has_setting<T, ICompute> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
