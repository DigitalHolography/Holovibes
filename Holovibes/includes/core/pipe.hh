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
#include "fast_updates_holder.hh"
#include "rendering.hh"
#include "converts.hh"
#include "postprocessing.hh"
#include "function_vector.hh"
#include "logger.hh"

#include "settings/settings.hh"
#include "settings/settings_container.hh"

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
     * \param stream The compute stream on which all the computations are processed.
     * \param settigns Default value for the settings of the pipe.
     */
    template <TupleContainsTypes<ALL_SETTINGS> InitSettings>
    Pipe(BatchInputQueue& input, Queue& output, Queue& record, const cudaStream_t& stream, InitSettings settings)
        : ICompute(input, output, record, stream, settings)
        , processed_output_fps_(FastUpdatesMap::map<FpsType>.create_entry(FpsType::OUTPUT_FPS))
    {
        ConditionType batch_condition = [&] { return batch_env_.batch_index == setting<settings::TimeStride>(); };

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
                                                                          moments_env_,
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
        set_requested(ICS::UpdateTimeTransformationSize, true);

        // Pre init
        if (setting<settings::FilterEnabled>())
            request(ICS::Filter);

        if (setting<settings::ConvolutionEnabled>())
            request(ICS::Convolution);
    }

    ~Pipe() override;

    /*! \brief Get the lens queue to display it. */
    std::unique_ptr<Queue>& get_lens_queue() override;

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

    /*! \brief Runs a function after the current pipe iteration ends */
    void insert_fn_end_vect(std::function<void()> function);

    /*! \brief Enqueue the main FunctionVector according to the requests. */
    void refresh() override;

    template <typename T>
    inline void update_setting(T setting)
    {
        LOG_TRACE("[Pipe] [update_setting] {}", typeid(T).name());

        if constexpr (has_setting_v<T, decltype(realtime_settings_)>)
            realtime_settings_.update_setting(setting);

        if constexpr (has_setting_v<T, decltype(onrestart_settings_)>)
            onrestart_settings_.update_setting(setting);

        if constexpr (has_setting_v<T, decltype(pipe_refresh_settings_)>)
            pipe_refresh_settings_.update_setting(setting);

        if constexpr (has_setting_v<T, compute::ImageAccumulation>)
            image_accumulation_->update_setting(setting);

        if constexpr (has_setting_v<T, compute::Rendering>)
            rendering_->update_setting(setting);

        if constexpr (has_setting_v<T, compute::FourierTransform>)
            fourier_transforms_->update_setting(setting);

        if constexpr (has_setting_v<T, compute::Converts>)
            converts_->update_setting(setting);

        if constexpr (has_setting_v<T, compute::Postprocessing>)
            postprocess_->update_setting(setting);
    }

  private:
    /*! \brief Make requests at the beginning of the refresh.
     *
     * Make the allocation of buffers when it is requested.
     *
     * \return return false if an allocation failed.
     */
    bool make_requests();

    /**
     * @brief Apply the updates of the settings on pipe refresh,
     */
    inline void pipe_refresh_apply_updates()
    {
        fourier_transforms_->pipe_refresh_apply_updates();
        image_accumulation_->pipe_refresh_apply_updates();
        pipe_refresh_settings_.apply_updates();
    }

    /*! \name Insert computation functions in the pipe
     * \{
     */
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

    void insert_moments();

    void insert_moments_record();

    void insert_input_to_moments();

    void insert_cuts_record();

    /*! \brief Reset the batch index if time_stride has been reached */
    void insert_reset_batch_index();
    /*! \}*/

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

    /*! \brief Enqueue a frame in an output queue
     *
     * \param output_queue Queue in which the frame is enqueued
     * \param frame Frame to enqueue
     * \param error Error message when an error occurs
     */
    void safe_enqueue_output(Queue& output_queue, unsigned short* frame, const std::string& error);

    /*! \brief Get the memcpy kind according to the setting type (usually a location setting).
     *
     * \param gpu Kind of memcpy for GPU
     * \param cpu Kind of memcpy for CPU
     * \return The kind of memcpy to use
     */
    template <typename SettingType>
    inline cudaMemcpyKind get_memcpy_kind(cudaMemcpyKind gpu = cudaMemcpyDeviceToDevice,
                                          cudaMemcpyKind cpu = cudaMemcpyDeviceToHost)
    {
        return setting<SettingType>() == Device::GPU ? gpu : cpu;
    }

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

    /*! \name Compute objects
     * \{
     */
    std::unique_ptr<compute::ImageAccumulation> image_accumulation_;
    std::unique_ptr<compute::FourierTransform> fourier_transforms_;
    std::unique_ptr<compute::Rendering> rendering_;
    std::unique_ptr<compute::Converts> converts_;
    std::unique_ptr<compute::Postprocessing> postprocess_;
    /*! \} */

    std::shared_ptr<std::atomic<unsigned int>> processed_output_fps_;
};
} // namespace holovibes

namespace holovibes
{
template <typename T>
struct has_setting<T, Pipe> : has_setting<T, ICompute>
{
};
} // namespace holovibes
