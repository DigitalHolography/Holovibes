/*! \file
 *
 * \brief #TODO Add a description for this file
 *
 * The Pipe is a sequential computing model, storing procedures
 * in a single container.
 */
#pragma once

#include "cuda_tools/unique_ptr.hh"
#include "icompute.hh"
#include "image_accumulation.hh"
#include "fourier_transform.hh"
#include "rendering.hh"
#include "converts.hh"
#include "postprocessing.hh"
#include "function_vector.hh"

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
    holovibes::settings::ReticleDisplayEnabled,    \
    holovibes::settings::Filter2dN1,               \
    holovibes::settings::Filter2dN2,               \
    holovibes::settings::Filter2dSmoothLow,        \
    holovibes::settings::Filter2dSmoothHigh,       \
    holovibes::settings::ChartRecordEnabled,       \
    holovibes::settings::FrameRecordEnabled,       \
    holovibes::settings::TimeTransformationSize,   \
    holovibes::settings::SpaceTransformation,      \
    holovibes::settings::TimeTransformation,       \
    holovibes::settings::Lambda,                   \
    holovibes::settings::ZDistance,                \
    holovibes::settings::ConvolutionEnabled,       \
    holovibes::settings::ConvolutionMatrix,        \
    holovibes::settings::DivideConvolutionEnabled, \
    holovibes::settings::ComputeMode,              \
    holovibes::settings::PixelSize,                \
    holovibes::settings::UnwrapHistorySize,        \
    holovibes::settings::SignalZone,               \
    holovibes::settings::NoiseZone,                \
    holovibes::settings::CompositeZone,            \
    holovibes::settings::TimeTransformationCutsOutputBufferSize, \
    holovibes::settings::CompositeKind,                          \
    holovibes::settings::CompositeAutoWeights,                   \
    holovibes::settings::RGB,                                    \
    holovibes::settings::HSV,                                    \
    holovibes::settings::ZFFTShift,                              \
    holovibes::settings::RecordFrameCount,                       \
    holovibes::settings::RecordMode


#define ONRESTART_SETTINGS                         \
    holovibes::settings::OutputBufferSize,         \
    holovibes::settings::RecordBufferSize,         \
    holovibes::settings::ContrastLowerThreshold,   \
    holovibes::settings::ContrastUpperThreshold,   \
    holovibes::settings::RenormConstant,           \
    holovibes::settings::CutsContrastPOffset,      \
    holovibes::settings::RecordQueueLocation,         \
    holovibes::settings::RawViewQueueLocation,        \
    holovibes::settings::InputQueueLocation

#define PIPEREFRESH_SETTINGS                       \
    holovibes::settings::TimeStride,               \
    holovibes::settings::BatchSize,                \
    holovibes::settings::InputFilter,              \
    holovibes::settings::FilterEnabled
 
#define ALL_SETTINGS REALTIME_SETTINGS, ONRESTART_SETTINGS, PIPEREFRESH_SETTINGS

// clang-format on

namespace holovibes
{
/*! \class Pipe
 *
 * \brief Pipe is a class that applies processing on input frames.
 *
 * # Why do it this way ?
 *
 * The goal of the pipe is to build a vector filled with functions to
 * apply on frames. This way it avoids to have a monolithic method plenty of
 * if/else following what the user wants to do. In most cases, the processing
 * remains the same at runtime, most jump conditions will always be the same.
 *
 * When the pipe is refreshed, the vector is updated with last user
 * parameters. Keep in mind that the software is incredibly faster than user
 * inputs in GUI, so treatments are always applied with the same parameters.
 *
 * ## RAII
 *
 * The pipe manages almost every CPU/GPU memory ressources. Once again,
 * most of frames buffer will always keep the same size, so it is not
 * necessary to allocate memory with malloc/cudaMalloc in each treatment
 * functions. Keep in mind, malloc is costly !
 *
 * ## Request system
 *
 * In order to avoid strange concurrent behaviours, the pipe is used with
 * a request system. When the compute descriptor is modified the GUI will
 * request the pipe to refresh with updated parameters.
 *
 * Also, some events such as autoconstrast will be executed only
 * for one iteration. For example, request_autocontrast will add the
 * autocontrast algorithm in the pipe and will automatically set a pipe refresh
 * so that the autocontrast algorithm will be done only once.
 */
class Pipe : public ICompute
{
  public:
    /*! \brief Allocate CPU/GPU ressources for computation.
     *
     * \param input Input queue containing acquired frames.
     * \param output Output queue where computed frames will be stored.
     * \param stream The compute stream on which all the computations are processed
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Pipe(BatchInputQueue& input, Queue& output, Queue& record, const cudaStream_t& stream, InitSettings settings)
        : ICompute(input, output, record, stream, settings)
        , realtime_settings_(settings)
        , onrestart_settings_(settings)
        , pipe_refresh_settings_(settings)
        , processed_output_fps_(GSH::fast_updates_map<FpsType>.create_entry(FpsType::OUTPUT_FPS))
    {
        ConditionType batch_condition = [&]() -> bool
        { return batch_env_.batch_index == setting<settings::TimeStride>(); };

        fn_compute_vect_ = FunctionVector(batch_condition);
        fn_end_vect_ = FunctionVector(batch_condition);

        image_accumulation_ = std::make_unique<compute::ImageAccumulation>(fn_compute_vect_,
                                                                           image_acc_env_,
                                                                           buffers_,
                                                                           input.get_fd(),
                                                                           stream_,
                                                                           settings);
        fourier_transforms_ = std::make_unique<compute::FourierTransform>(fn_compute_vect_,
                                                                          buffers_,
                                                                          input.get_fd(),
                                                                          spatial_transformation_plan_,
                                                                          time_transformation_env_,
                                                                          stream_,
                                                                          settings);
        rendering_ = std::make_unique<compute::Rendering>(fn_compute_vect_,
                                                          buffers_,
                                                          chart_env_,
                                                          image_acc_env_,
                                                          time_transformation_env_,
                                                          input.get_fd(),
                                                          output.get_fd(),
                                                          stream_,
                                                          settings);
        converts_ = std::make_unique<compute::Converts>(fn_compute_vect_,
                                                        buffers_,
                                                        time_transformation_env_,
                                                        plan_unwrap_2d_,
                                                        input.get_fd(),
                                                        stream_,
                                                        settings);
        postprocess_ =
            std::make_unique<compute::Postprocessing>(fn_compute_vect_, buffers_, input.get_fd(), stream_, settings);

        *processed_output_fps_ = 0;
        update_time_transformation_size_requested_ = true;

        pre_init();

        try
        {
            refresh();
        }
        catch (const holovibes::CustomException& e)
        {
            // If refresh() fails the compute descriptor settings will be
            // changed to something that should make refresh() work
            // (ex: lowering the GPU memory usage)
            LOG_WARN("Pipe refresh failed, trying one more time with updated compute descriptor");
            LOG_WARN("Exception: {}", e.what());
            try
            {
                refresh();
            }
            catch (const holovibes::CustomException& e)
            {
                // If it still didn't work holovibes is probably going to freeze
                // and the only thing you can do is restart it manually
                LOG_ERROR("Pipe could not be initialized, You might want to restart holovibes");
                LOG_ERROR("Exception: {}", e.what());
                throw e;
            }
        }
    }

    ~Pipe() override;

    /*! \brief Get the lens queue to display it. */
    std::unique_ptr<Queue>& get_lens_queue() override;

    /*! \brief Runs a function after the current pipe iteration ends */
    void insert_fn_end_vect(std::function<void()> function);

    /*! \brief Execute one processing iteration.
     *
     * Checks the number of frames in input queue, that must at least be 1.
     * Call each function stored in the FunctionVector.
     * Call each function stored in the end FunctionVector, then clears it
     * Enqueue the output frame contained in gpu_output_buffer.
     * Dequeue one frame of the input queue.
     * Check if a ICompute refresh has been requested.
     */
    void exec() override;

    void pre_init()
    {
        if (setting<settings::FilterEnabled>())
            request_filter();

        if (setting<settings::ConvolutionEnabled>())
            request_convolution();
    }

    /*! \brief Enqueue the main FunctionVector according to the requests. */
    void refresh() override;

    /**
     * @brief Contains all the settings of the worker that should be updated
     * on restart.
     */
    RealtimeSettingsContainer<REALTIME_SETTINGS> realtime_settings_;
    DelayedSettingsContainer<ONRESTART_SETTINGS> onrestart_settings_;
    DelayedSettingsContainer<PIPEREFRESH_SETTINGS> pipe_refresh_settings_;

    template <typename T>
    inline void update_setting(T setting)
    {
        spdlog::trace("[Pipe] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting<T, decltype(realtime_settings_)>::value)
        {
            realtime_settings_.update_setting(setting);
        }

        if constexpr (has_setting<T, decltype(onrestart_settings_)>::value)
        {
            onrestart_settings_.update_setting(setting);
        }

        if constexpr (has_setting<T, decltype(pipe_refresh_settings_)>::value)
        {
            pipe_refresh_settings_.update_setting(setting);
        }

        if constexpr (has_setting<T, compute::ImageAccumulation>::value)
        {
            image_accumulation_->update_setting(setting);
        }
        if constexpr (has_setting<T, compute::Rendering>::value)
        {
            rendering_->update_setting(setting);
        }
        if constexpr (has_setting<T, compute::FourierTransform>::value)
        {
            fourier_transforms_->update_setting(setting);
        }
        if constexpr (has_setting<T, compute::Converts>::value)
        {
            converts_->update_setting(setting);
        }
        if constexpr (has_setting<T, compute::Postprocessing>::value)
        {
            postprocess_->update_setting(setting);
        }
        if constexpr (has_setting<T, ICompute>::value)
        {
            update_setting_icompute(setting);
        }
    }

    inline void on_restart_apply_settings() { onrestart_settings_.apply_updates(); }

    /**
     * @brief Apply the updates of the settings on pipe refresh,
     */
    inline void pipe_refresh_apply_updates()
    {
        fourier_transforms_->pipe_refresh_apply_updates();
        icompute_pipe_refresh_apply_updates();
        pipe_refresh_settings_.apply_updates();
    }

  protected:
    /*! \brief Make requests at the beginning of the refresh.
     *
     * Make the allocation of buffers when it is requested.
     *
     * \return return false if an allocation failed.
     */
    bool make_requests();

    /*! \brief Transfer from gpu_space_transformation_buffer to gpu_time_transformation_queue for time transform */
    void insert_transfer_for_time_transformation();

    /*! \brief Increase batch_index by batch_size */
    void update_batch_index();

    /*! \brief Wait that there are at least a batch of frames in input queue */
    void insert_wait_frames();

    /*! \brief Dequeue the input queue frame by frame in raw mode */
    void insert_dequeue_input();

    /*! \brief Enqueue the output frame in the output queue in hologram mode */
    void insert_output_enqueue_hologram_mode();

    /*! \brief Enqueue the output frame in the filter2d view queue */
    void insert_filter2d_view();

    /*! \brief Request the computation of a autocontrast if the contrast and the contrast refresh is enabled */
    void insert_request_autocontrast();

    void insert_raw_view();

    void insert_raw_record();

    void insert_hologram_record();

    void insert_cuts_record();

    /*! \brief Reset the batch index if time_stride has been reached */
    void insert_reset_batch_index();

    /*! \brief Enqueue a frame in an output queue
     *
     * \param output_queue Queue in which the frame is enqueued
     * \param frame Frame to enqueue
     * \param error Error message when an error occurs
     */
    void safe_enqueue_output(Queue& output_queue, unsigned short* frame, const std::string& error);

  private:
    /*! \brief Vector of functions that will be executed in the exec() function. */
    FunctionVector fn_compute_vect_;

    /*! \brief Vecor of functions that will be executed once, after the execution of fn_compute_vect_. */
    FunctionVector fn_end_vect_;

    /*! \brief Mutex that prevents the insertion of a function during its execution.
     *
     * Since we can insert functions in fn_end_vect_ from other threads  MainWindow), we need to lock it.
     */
    std::mutex fn_end_vect_mutex_;

    std::unique_ptr<compute::ImageAccumulation> image_accumulation_;
    std::unique_ptr<compute::FourierTransform> fourier_transforms_;
    std::unique_ptr<compute::Rendering> rendering_;
    std::unique_ptr<compute::Converts> converts_;
    std::unique_ptr<compute::Postprocessing> postprocess_;

    std::shared_ptr<std::atomic<unsigned int>> processed_output_fps_;

    /*! \brief Iterates and executes function of the pipe.
     *
     * It will first iterate over fn_compute_vect_, then over function_end_pipe_.
     */
    void run_all();

    /*! \brief Force contiguity on record queue when cli is active.
     *
     * \param nb_elm_to_add the number of elements that might be added in the record queue
     */
    void keep_contiguous(int nb_elm_to_add) const;

    /*! \brief Updates all attribute caches with the reference held by GSH */
    void synchronize_caches();

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
struct has_setting<T, Pipe> : is_any_of<T, ALL_SETTINGS>
{
};
} // namespace holovibes