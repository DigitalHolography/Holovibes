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
#include "compute_env.hh"
#include "concurrent_deque.hh"
#include "enum_window_kind.hh"
#include "enum_record_mode.hh"
#include "logger.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

#define ICS holovibes::ICompute::Setting

#pragma region Settings configuration

// clang-format off
#define REALTIME_SETTINGS                                        \
    holovibes::settings::InputFPS,                               \
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
    holovibes::settings::ReticleDisplayEnabled,                  \
    holovibes::settings::Filter2dN1,                             \
    holovibes::settings::Filter2dN2,                             \
    holovibes::settings::Filter2dSmoothLow,                      \
    holovibes::settings::Filter2dSmoothHigh,                     \
    holovibes::settings::ChartRecordEnabled,                     \
    holovibes::settings::FrameRecordEnabled,                     \
    holovibes::settings::TimeTransformationSize,                 \
    holovibes::settings::SpaceTransformation,                    \
    holovibes::settings::TimeTransformation,                     \
    holovibes::settings::Lambda,                                 \
    holovibes::settings::ZDistance,                              \
    holovibes::settings::ConvolutionEnabled,                     \
    holovibes::settings::ConvolutionMatrix,                      \
    holovibes::settings::DivideConvolutionEnabled,               \
    holovibes::settings::ComputeMode,                            \
    holovibes::settings::PixelSize,                              \
    holovibes::settings::SignalZone,                             \
    holovibes::settings::NoiseZone,                              \
    holovibes::settings::CompositeZone,                          \
    holovibes::settings::TimeTransformationCutsOutputBufferSize, \
    holovibes::settings::CompositeKind,                          \
    holovibes::settings::CompositeAutoWeights,                   \
    holovibes::settings::RGB,                                    \
    holovibes::settings::HSV,                                    \
    holovibes::settings::ZFFTShift,                              \
    holovibes::settings::RecordFrameCount,                       \
    holovibes::settings::RecordMode,                             \
    holovibes::settings::TimeWindow,                             \
    holovibes::settings::ArteryMaskEnabled,                      \
    holovibes::settings::OtsuEnabled,                            \
    holovibes::settings::VesselnessSigma


#define ONRESTART_SETTINGS                                       \
    holovibes::settings::OutputBufferSize,                       \
    holovibes::settings::RecordBufferSize,                       \
    holovibes::settings::ContrastLowerThreshold,                 \
    holovibes::settings::ContrastUpperThreshold,                 \
    holovibes::settings::RenormConstant,                         \
    holovibes::settings::CutsContrastPOffset,                    \
    holovibes::settings::RecordQueueLocation,                    \
    holovibes::settings::RawViewQueueLocation,                   \
    holovibes::settings::InputQueueLocation

#define PIPEREFRESH_SETTINGS                                     \
    holovibes::settings::TimeStride,                             \
    holovibes::settings::BatchSize,                              \
    holovibes::settings::XY,                                     \
    holovibes::settings::XZ,                                     \
    holovibes::settings::YZ,                                     \
    holovibes::settings::InputFilter,                            \
    holovibes::settings::FilterEnabled,                          \
    holovibes::settings::DataType

#define ALL_SETTINGS REALTIME_SETTINGS, ONRESTART_SETTINGS, PIPEREFRESH_SETTINGS

// clang-format on
#pragma endregion

namespace holovibes
{
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
        , onrestart_settings_(settings)
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

        fd.depth = camera::PixelDepth::Complex;
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

        UpdateTimeTransformationAlgorithm,

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
    std::unique_ptr<Queue>& get_stft_slice_queue(int slice)
    {
        return slice ? time_transformation_env_.gpu_output_queue_yz : time_transformation_env_.gpu_output_queue_xz;
    }

    virtual std::unique_ptr<Queue>& get_lens_queue() = 0;

    std::unique_ptr<Queue>& get_raw_view_queue() { return gpu_raw_view_queue_; };

    std::unique_ptr<Queue>& get_filter2d_view_queue() { return gpu_filter2d_view_queue_; };

    std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_display_queue() { return chart_env_.chart_display_queue_; };

    std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_record_queue() { return chart_env_.chart_record_queue_; }
    /*! \} */

  protected:
    virtual void refresh() = 0;

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
     *
     * The function modifies the buffers f0_buffer, f1_buffer and f2_buffer in ICompute
     */
    void fft_freqs();

    /*!
     * \brief Resize all the buffers using the `time_transformation_size` and recaclulate the `fft_freqs` for the
     * moments.
     *
     * \param  time_transformation_size  The new time transformation size.
     * \return                           Whether there is an error or not.
     */
    virtual bool update_time_transformation_size(const unsigned short time_transformation_size);

    /*! \name Resources management
     * \{
     */
    void update_spatial_transformation_parameters();

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

    /**
     * @brief Helper function to get a settings value.
     */
    template <typename T>
    auto setting()
    {
        if constexpr (has_setting_v<T, decltype(realtime_settings_)>)
            return realtime_settings_.get<T>().value;

        if constexpr (has_setting_v<T, decltype(onrestart_settings_)>)
            return onrestart_settings_.get<T>().value;

        if constexpr (has_setting_v<T, decltype(pipe_refresh_settings_)>)
            return pipe_refresh_settings_.get<T>().value;
    }

    /*! \brief Performs tasks specific to the current time transformation setting.
     *  \param size The size for time transformation.
     */
    void perform_time_transformation_setting_specific_tasks(const unsigned short size);

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

    /*! \brief Moments environment. */
    MomentsEnv moments_env_;

    /*! \brief Chart environment. */
    ChartEnv chart_env_;

    /*! \brief Image accumulation environment */
    ImageAccEnv image_acc_env_;

    /*! \brief Vesselness masks environment. */
    VesselnessMaskEnv vesselness_mask_env_;

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

    /**
     * @brief Contains all the settings of the worker that should be updated
     * on restart.
     */
    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
    /*! \} */

  private:
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
