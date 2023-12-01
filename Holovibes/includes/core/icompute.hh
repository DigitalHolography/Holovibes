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

#pragma region Settings configuration
// clang-format off

#define REALTIME_SETTINGS                          \
    holovibes::settings::ImageType,                \
    holovibes::settings::X,                        \
    holovibes::settings::Y,                        \
    holovibes::settings::P,                        \
    holovibes::settings::Q,                        \
    holovibes::settings::XY,                       \
    holovibes::settings::XZ,                       \
    holovibes::settings::YZ,                       \
    holovibes::settings::Filter2d,                 \
    holovibes::settings::CurrentWindow,            \
    holovibes::settings::LensViewEnabled,          \
    holovibes::settings::ChartDisplayEnabled,      \
    holovibes::settings::Filter2dEnabled,          \
    holovibes::settings::Filter2dViewEnabled,      \
    holovibes::settings::FftShiftEnabled,          \
    holovibes::settings::RawViewEnabled,           \
    holovibes::settings::CutsViewEnabled,          \
    holovibes::settings::RenormEnabled,            \
    holovibes::settings::ReticleScale,             \
    holovibes::settings::Filter2dN1,               \
    holovibes::settings::Filter2dN2,               \
    holovibes::settings::Filter2dSmoothLow,        \
    holovibes::settings::Filter2dSmoothHigh,       \
    holovibes::settings::TimeTransformationSize,   \
    holovibes::settings::TimeTransformation,       \
    holovibes::settings::TimeTransformationCutsOutputBufferSize

#define PIPEREFRESH_SETTINGS                         \
    holovibes::settings::BatchSize

#define ALL_SETTINGS REALTIME_SETTINGS, PIPEREFRESH_SETTINGS

// clang-format on

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
    /*! \brief Float XZ buffer of 1 frame, filled with the correct computer p XZ frame. */
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
     * The variable is incremented until it reachs batch_size in
     *enqueue_multiple, then it is set back to 0
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
    /*! \brief STFT buffer.  Contains the result of the STFT done on the STFT queue.
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

/*! \struct FrameRecordEnv
 *
 * \brief #TODO Add a description for this struct
 */
struct FrameRecordEnv
{
    std::unique_ptr<Queue> frame_record_queue_ = nullptr;
    std::atomic<RecordMode> record_mode_{RecordMode::NONE};
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
// #TODO Add \name tags between groups of methods and attributes to make the documentation clearer
class ICompute
{
  public:
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    ICompute(BatchInputQueue& input, Queue& output, const cudaStream_t& stream, InitSettings settings)
        : gpu_input_queue_(input)
        , gpu_output_queue_(output)
        , stream_(stream)
        , past_time_(std::chrono::high_resolution_clock::now())
        , realtime_settings_(settings)
        , pipe_refresh_settings_(settings)
    {
        int err = 0;

        plan_unwrap_2d_.plan(gpu_input_queue_.get_fd().width, gpu_input_queue_.get_fd().height, CUFFT_C2C);

        const camera::FrameDescriptor& fd = gpu_input_queue_.get_fd();
        long long int n[] = {fd.height, fd.width};

        // This plan has a useful significant memory cost, check XtplanMany comment
        spatial_transformation_plan_.XtplanMany(2, // 2D
                                                n, // Dimension of inner most & outer most dimension
                                                n, // Storage dimension size
                                                1, // Between two inputs (pixels) of same image distance is one
                                                fd.get_frame_res(), // Distance between 2 same index pixels of 2 images
                                                CUDA_C_32F,         // Input type
                                                n,
                                                1,
                                                fd.get_frame_res(),             // Ouput layout same as input
                                                CUDA_C_32F,                     // Output type
                                                setting<settings::BatchSize>(), // Batch size
                                                CUDA_C_32F);                    // Computation type

        int inembed[1];
        int zone_size = static_cast<int>(gpu_input_queue_.get_fd().get_frame_res());

        inembed[0] = setting<settings::TimeTransformationSize>();

        time_transformation_env_.stft_plan
            .planMany(1, inembed, inembed, zone_size, 1, inembed, zone_size, 1, CUFFT_C2C, zone_size);

        camera::FrameDescriptor new_fd = gpu_input_queue_.get_fd();
        new_fd.depth = 8;
        // FIXME-CAMERA : WTF depth 8 ==> maybe a magic value for complex mode
        time_transformation_env_.gpu_time_transformation_queue.reset(
            new Queue(new_fd, setting<settings::TimeTransformationSize>()));

        // Static cast size_t to avoid overflow
        if (!buffers_.gpu_spatial_transformation_buffer.resize(
                static_cast<const size_t>(setting<settings::BatchSize>()) * gpu_input_queue_.get_fd().get_frame_res()))
            err++;

        int output_buffer_size = gpu_input_queue_.get_fd().get_frame_res();
        if (setting<settings::ImageType>() == ImgType::Composite)
            image::grey_to_rgb_size(output_buffer_size);
        if (!buffers_.gpu_output_frame.resize(output_buffer_size))
            err++;
        buffers_.gpu_postprocess_frame_size = static_cast<int>(gpu_input_queue_.get_fd().get_frame_res());

        if (setting<settings::ImageType>() == ImgType::Composite)
            image::grey_to_rgb_size(buffers_.gpu_postprocess_frame_size);

        if (!buffers_.gpu_postprocess_frame.resize(buffers_.gpu_postprocess_frame_size))
            err++;

        // Init the gpu_p_frame with the size of input image
        if (!time_transformation_env_.gpu_p_frame.resize(buffers_.gpu_postprocess_frame_size))
            err++;

        if (!buffers_.gpu_complex_filter2d_frame.resize(buffers_.gpu_postprocess_frame_size))
            err++;

        if (!buffers_.gpu_float_filter2d_frame.resize(buffers_.gpu_postprocess_frame_size))
            err++;

        if (!buffers_.gpu_filter2d_frame.resize(buffers_.gpu_postprocess_frame_size))
            err++;

        if (!buffers_.gpu_filter2d_mask.resize(output_buffer_size))
            err++;

        if (!buffers_.gpu_input_filter_mask.resize(output_buffer_size))
            err++;

        if (err != 0)
            throw std::exception(cudaGetErrorString(cudaGetLastError()));
    }

    template <typename T>
    inline void update_setting_icompute(T setting)
    {
        spdlog::info("[ICompute] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            realtime_settings_.update_setting(setting);
        }

        if constexpr (has_setting<T, decltype(pipe_refresh_settings_)>::value)
        {
            pipe_refresh_settings_.update_setting(setting);
        }
    }

    inline void icompute_pipe_refresh_apply_updates() {
        pipe_refresh_settings_.apply_updates();
    }
    // #TODO Check if soft_request_refresh is even needed or if request_refresh is enough in MainWindow
    void soft_request_refresh();
    void request_refresh();
    void request_output_resize(unsigned int new_output_size);
    void request_autocontrast(WindowKind kind);
    void request_update_time_transformation_size();
    void request_unwrapping_1d(const bool value);
    void request_unwrapping_2d(const bool value);
    void request_display_chart();
    void request_disable_display_chart();
    void request_record_chart(unsigned int nb_chart_points_to_record);
    void request_disable_record_chart();
    void request_termination();
    void request_update_batch_size();
    void request_update_time_stride();
    void request_disable_lens_view();
    void request_raw_view();
    void request_disable_raw_view();
    void request_filter2d_view();
    void request_disable_filter2d_view();
    void request_hologram_record();
    void request_raw_record();
    void request_cuts_record(RecordMode rm);
    void request_disable_frame_record();
    void request_clear_img_acc();
    void request_convolution();
    void request_filter();
    void request_disable_convolution();
    void request_disable_filter();

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

    void create_stft_slice_queue();
    void delete_stft_slice_queue();
    std::unique_ptr<Queue>& get_stft_slice_queue(int i);
    bool get_cuts_request();
    bool get_cuts_delete_request();

    bool get_unwrap_1d_request() const { return unwrap_1d_requested_; }
    bool get_unwrap_2d_request() const { return unwrap_2d_requested_; }
    bool get_autocontrast_request() const { return autocontrast_requested_; }
    bool get_autocontrast_slice_xz_request() const { return autocontrast_slice_xz_requested_; }
    bool get_autocontrast_slice_yz_request() const { return autocontrast_slice_yz_requested_; }
    bool get_refresh_request() const { return refresh_requested_; }
    bool get_update_time_transformation_size_request() const { return update_time_transformation_size_requested_; }
    bool get_stft_update_roi_request() const { return stft_update_roi_requested_; }
    bool get_termination_request() const { return termination_requested_; }
    bool get_request_time_transformation_cuts() const { return request_time_transformation_cuts_; }
    bool get_request_delete_time_transformation_cuts() const { return request_delete_time_transformation_cuts_; }
    std::optional<unsigned int> get_output_resize_request() const { return output_resize_requested_; }
    bool get_raw_view_requested() const { return raw_view_requested_; }
    bool get_disable_raw_view_requested() const { return disable_raw_view_requested_; }
    bool get_disable_lens_view_requested() const { return request_disable_lens_view_; }
    bool get_filter2d_view_requested() const { return filter2d_view_requested_; }
    bool get_disable_filter2d_view_requested() const { return disable_filter2d_view_requested_; }
    bool get_chart_display_requested() const { return chart_display_requested_; }
    std::optional<unsigned int> get_chart_record_requested() const { return chart_record_requested_; }
    bool get_disable_chart_display_requested() const { return disable_chart_display_requested_; }
    bool get_disable_chart_record_requested() const { return disable_chart_record_requested_; }
    bool get_hologram_record_requested() const { return hologram_record_requested_; }
    bool get_raw_record_requested() const { return raw_record_requested_; }
    bool get_cuts_record_requested() const { return cuts_record_requested_; }
    bool get_disable_frame_record_requested() const { return disable_frame_record_requested_; }
    bool get_convolution_requested() const { return convolution_requested_; }
    bool get_disable_convolution_requested() const { return convolution_requested_; }
    bool get_filter_requested() const { return filter_requested_; }
    bool get_disable_filter_requested() const { return filter_requested_; }

    virtual std::unique_ptr<Queue>& get_lens_queue() = 0;

    virtual std::unique_ptr<Queue>& get_raw_view_queue();

    virtual std::unique_ptr<Queue>& get_filter2d_view_queue();

    virtual std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_display_queue();

    virtual std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_record_queue();

    virtual std::unique_ptr<Queue>& get_frame_record_queue();

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

    ICompute& operator=(const ICompute&) = delete;
    ICompute(const ICompute&) = delete;
    virtual ~ICompute() {}

  protected:
    /*! \brief Reference on the input queue */
    BatchInputQueue& gpu_input_queue_;

    /*! \brief Reference on the output queue */
    Queue& gpu_output_queue_;

    /*! \brief Main buffers. */
    CoreBuffersEnv buffers_;

    /*! \brief Batch environment */
    BatchEnv batch_env_;

    /*! \brief STFT environment. */
    TimeTransformationEnv time_transformation_env_;

    /*! \brief Frame Record environment (Raw + Hologram + Cuts) */
    FrameRecordEnv frame_record_env_;

    /*! \brief Chart environment. */
    ChartEnv chart_env_;

    /*! \brief Image accumulation environment */
    ImageAccEnv image_acc_env_;

    /*! \brief Queue storing raw frames used by raw view */
    std::unique_ptr<Queue> gpu_raw_view_queue_{nullptr};

    /*! \brief Queue storing filter2d frames */
    std::unique_ptr<Queue> gpu_filter2d_view_queue_{nullptr};

    /*! \brief Pland 2D. Used for spatial fft performed on the complex input frame. */
    cuda_tools::CufftHandle spatial_transformation_plan_;

    /*! \brief Pland 2D. Used for unwrap 2D. */
    cuda_tools::CufftHandle plan_unwrap_2d_;

    /*! \brief Compute stream to perform pipe computation */
    const cudaStream_t& stream_;

    /*! \brief Chrono counting time between two iteration
     *
     * Taking into account steps, since it is executing at the end of pipe.
     */
    /* FIXME: not used anywhere */
    std::chrono::time_point<std::chrono::steady_clock> past_time_;

    /*! \brief Counting pipe iteration, in order to update fps only every 100 iterations. */
    unsigned int frame_count_{0};

    std::atomic<bool> unwrap_1d_requested_{false};
    std::atomic<bool> unwrap_2d_requested_{false};
    std::atomic<bool> autocontrast_requested_{false};
    std::atomic<bool> autocontrast_slice_xz_requested_{false};
    std::atomic<bool> autocontrast_slice_yz_requested_{false};
    std::atomic<bool> autocontrast_filter2d_requested_{false};
    std::atomic<bool> refresh_requested_{false};
    std::atomic<bool> update_time_transformation_size_requested_{false};
    std::atomic<bool> stft_update_roi_requested_{false};
    std::atomic<bool> chart_display_requested_{false};
    std::atomic<bool> disable_chart_display_requested_{false};
    std::atomic<std::optional<unsigned int>> chart_record_requested_{std::nullopt};
    std::atomic<bool> disable_chart_record_requested_{false};
    std::atomic<std::optional<unsigned int>> output_resize_requested_{std::nullopt};
    std::atomic<bool> raw_view_requested_{false};
    std::atomic<bool> disable_raw_view_requested_{false};
    std::atomic<bool> filter2d_view_requested_{false};
    std::atomic<bool> disable_filter2d_view_requested_{false};
    std::atomic<bool> termination_requested_{false};
    std::atomic<bool> request_time_transformation_cuts_{false};
    std::atomic<bool> request_delete_time_transformation_cuts_{false};
    std::atomic<bool> request_update_batch_size_{false};
    std::atomic<bool> request_update_time_stride_{false};
    std::atomic<bool> request_disable_lens_view_{false};
    std::atomic<bool> hologram_record_requested_{false};
    std::atomic<bool> raw_record_requested_{false};
    std::atomic<bool> cuts_record_requested_{false};
    std::atomic<bool> disable_frame_record_requested_{false};
    std::atomic<bool> request_clear_img_accu{false};
    std::atomic<bool> convolution_requested_{false};
    std::atomic<bool> disable_convolution_requested_{false};
    std::atomic<bool> filter_requested_{false};
    std::atomic<bool> disable_filter_requested_{false};

    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
    DelayedSettingsContainer<PIPEREFRESH_SETTINGS> pipe_refresh_settings_;

    CompositeCache::Cache composite_cache_;

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
        if constexpr (has_setting<T, decltype(pipe_refresh_settings_)>::value)
        {
            return pipe_refresh_settings_.get<T>().value;
        }
    }
};
} // namespace holovibes

namespace holovibes
{
template <typename T>
struct has_setting<T, ICompute> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes
