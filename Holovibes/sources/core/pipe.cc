#include "pipe.hh"

#include "queue.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "logger.hh"

#include "filter2D.cuh"
#include "stft.cuh"
#include "convolution.cuh"
#include "composite.cuh"
#include "tools.cuh"
#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "tools.hh"
#include "contrast_correction.cuh"
#include "enqueue_exception.hh"
#include "aliases.hh"
#include "holovibes.hh"
#include "cuda_memory.cuh"
#include "fast_updates_holder.hh"

#include "API.hh"

namespace holovibes
{

bool Pipe::can_insert_to_record_queue(int nb_elm_to_add)
{
    // When stopping a record the record queue is emptied and the FrameAcquisitionEnabled setting is set to false.
    // But the pipe isn't refreshed directly so the insert_XXX_function still insert in the record queue.
    if (!setting<settings::FrameAcquisitionEnabled>())
        return false;

    bool unlimited_record = setting<settings::RecordFrameCount>() == std::nullopt;

    if (record_queue_.has_overwritten() || input_queue_.has_overwritten())
    {
        API.record.set_frame_acquisition_enabled(false);
        total_nb_frames_to_acquire_ = nb_frames_acquired_.load();
        return false;
    }

    size_t total =
        total_nb_frames_to_acquire_ * (setting<settings::FrameSkip>() + 1) + setting<settings::RecordFrameOffset>();

    if (!unlimited_record && nb_frames_acquired_ >= total)
    {
        API.record.set_frame_acquisition_enabled(false);
        return false;
    }

    // This loop might be useless since it's an > and not a >= so the record queue will be overwriten and the record
    // will stop
    while (API.record.is_recording() && record_queue_.get_size() + nb_elm_to_add > record_queue_.get_max_size())
        continue;

    nb_frames_acquired_ += nb_elm_to_add;
    if (unlimited_record)
        total_nb_frames_to_acquire_ += nb_elm_to_add;

    return true;
}

using camera::FrameDescriptor;

Pipe::~Pipe()
{
    FastUpdatesMap::map<IntType>.remove_entry(IntType::OUTPUT_FPS);
    FastUpdatesMap::map<RecordType>.remove_entry(RecordType::FRAME);
}

#define HANDLE_REQUEST(setting, log_message, action)                                                                   \
    if (is_requested(setting))                                                                                         \
    {                                                                                                                  \
        LOG_DEBUG(log_message "requested");                                                                            \
        action;                                                                                                        \
        clear_request(setting);                                                                                        \
    }

bool Pipe::make_requests()
{
    // In order to have a better memory management, free all the ressources that needs to be freed first and allocate
    // the ressources that need to beallocated in second
    bool success_allocation = true;
    auto& api = API;

    /* Free buffers */
    HANDLE_REQUEST(ICS::DisableConvolution, "Disable convolution", postprocess_->dispose());

    HANDLE_REQUEST(ICS::DisableLensView, "Disable lens view", fourier_transforms_->get_lens_queue().reset(nullptr));

    if (is_requested(ICS::DisableRawView))
    {
        LOG_DEBUG("disable_raw_view_requested");

        gpu_raw_view_queue_.reset(nullptr);
        clear_request(ICS::DisableRawView);
    }

    if (is_requested(ICS::DisableFilter2DView))
    {
        LOG_DEBUG("disable_filter2D_view_requested");

        gpu_filter2d_view_queue_.reset(nullptr);
        api.view.set_filter2d_view_enabled(false);
        clear_request(ICS::DisableFilter2DView);
    }

    if (is_requested(ICS::DeleteTimeTransformationCuts))
    {
        LOG_DEBUG("Delete time transformation cuts");

        dispose_cuts();
        image_accumulation_->dispose_cuts_queue();

        clear_request(ICS::DeleteTimeTransformationCuts);
    }

    if (is_requested(ICS::DisableChartDisplay))
    {
        LOG_DEBUG("disable_chart_display_requested");

        chart_env_.chart_display_queue_.reset(nullptr);
        api.view.set_chart_display_enabled(false);
        clear_request(ICS::DisableChartDisplay);
    }

    if (is_requested(ICS::DisableChartRecord))
    {
        LOG_DEBUG("disable_chart_record_requested");

        chart_env_.chart_record_queue_.reset(nullptr);
        api.record.set_chart_record_enabled(false);
        chart_env_.nb_chart_points_to_record_ = 0;
        nb_frames_acquired_ = 0;
        clear_request(ICS::DisableChartRecord);
    }

    if (is_requested(ICS::DisableFrameRecord))
    {
        LOG_DEBUG("disable_frame_record_requested");

        api.record.set_frame_acquisition_enabled(false);
        total_nb_frames_to_acquire_ = nb_frames_acquired_.load();
        clear_request(ICS::DisableFrameRecord);
    }

    image_accumulation_->dispose(); // done only if requested

    /* Allocate buffer */
    HANDLE_REQUEST(ICS::UpdateTimeTransformationAlgorithm,
                   "Update time tr. algorithm",
                   perform_time_transformation_setting_specific_tasks(setting<settings::TimeTransformationSize>()));

    if (is_requested(ICS::OutputBuffer))
    {
        LOG_DEBUG("output_buffer_requested");
        init_output_queue();
        clear_request(ICS::OutputBuffer);
    }

    HANDLE_REQUEST(ICS::LensView, "Allocate lens view", fourier_transforms_->init_lens_queue());

    HANDLE_REQUEST(ICS::Convolution, "Convolution", postprocess_->init());

    // Updating number of images
    if (is_requested(ICS::UpdateTimeTransformationSize))
    {
        LOG_DEBUG("update_time_transformation_size_requested");

        if (!update_time_transformation_size(setting<settings::TimeTransformationSize>()))
        {
            success_allocation = false;
            api.transform.set_p_index(0);
            api.transform.set_time_transformation_size(1);
            update_time_transformation_size(1);
            LOG_WARN("Updating #img failed; #img updated to 1");
        }

        clear_request(ICS::UpdateTimeTransformationSize);
    }

    HANDLE_REQUEST(ICS::UpdateTimeStride, "Update time stride", batch_env_.batch_index = 0);

    if (is_requested(ICS::UpdateBatchSize))
    {
        LOG_DEBUG("request_update_batch_size");

        update_spatial_transformation_parameters();
        input_queue_.resize(setting<settings::BatchSize>());
        clear_request(ICS::UpdateBatchSize);
    }

    if (is_requested(ICS::TimeTransformationCuts))
    {
        LOG_DEBUG("Time transformation cuts");

        init_cuts();
        image_accumulation_->init_cuts_queue();

        clear_request(ICS::TimeTransformationCuts);
    }

    image_accumulation_->init(); // done only if requested
    image_accumulation_->clear();

    if (is_requested(ICS::RawView))
    {
        LOG_DEBUG("raw_view_requested");

        auto fd = input_queue_.get_fd();
        gpu_raw_view_queue_.reset(new Queue(fd, static_cast<unsigned int>(setting<settings::OutputBufferSize>())));
        clear_request(ICS::RawView);
    }

    if (is_requested(ICS::Filter2DView))
    {
        LOG_DEBUG("filter2d_view_requested");

        auto fd = buffers_.gpu_output_queue->get_fd();
        gpu_filter2d_view_queue_.reset(new Queue(fd, static_cast<unsigned int>(setting<settings::OutputBufferSize>())));
        api.view.set_filter2d_view_enabled(true);
        clear_request(ICS::Filter2DView);
    }

    if (is_requested(ICS::ChartDisplay))
    {
        LOG_DEBUG("chart_display_requested");

        chart_env_.chart_display_queue_.reset(new ConcurrentDeque<ChartPoint>());
        api.view.set_chart_display_enabled(true);
        clear_request(ICS::ChartDisplay);
    }

    if (chart_record_requested_.load() != std::nullopt)
    {
        LOG_DEBUG("chart_record_requested");

        chart_env_.chart_record_queue_.reset(new ConcurrentDeque<ChartPoint>());
        api.record.set_chart_record_enabled(true);
        chart_env_.nb_chart_points_to_record_ = chart_record_requested_.load().value();
        chart_record_requested_ = std::nullopt;
    }

    if (is_requested(ICS::UpdateRegistrationZone))
    {
        registration_->updade_cirular_mask();
        clear_request(ICS::UpdateRegistrationZone);
    }

    return success_allocation;
}

void Pipe::refresh()
{
    pipe_refresh_apply_updates();
    clear_request(ICS::Refresh);

    fn_compute_vect_->clear();

    // Aborting if allocation failed
    if (!make_requests())
    {
        clear_request(ICS::Refresh);
        return;
    }

    apply_realtime_settings();
    pipe_refresh_apply_updates();

    /*
     * With the --default-stream per-thread nvcc options, each thread runs cuda
     * calls/operations on its own default stream. Cuda calls/operations ran on
     * different default streams are processed concurrently by the GPU.
     *
     * Thus, the FrameReadWorker and the ComputeWorker run concurrently on the
     * CPU and in addition concurrently on the device which leads to a higher
     * GPU usage.
     *
     * WARNING: All cuda calls/operations on the thread compute are asynchronous
     * with respect to the host. Only one stream synchronisation
     * (cudaStreamSynchronize) at the end of the pipe is required. Adding
     * more stream synchronisation is not needed.
     * Using cudaDeviceSynchronize is FORBIDDEN as it will synchronize this
     * thread stream with all other streams (reducing performances drastically
     * because of a longer waiting time).
     *
     * The value of the default stream is 0. However, not providing a stream to
     * cuda calls forces the default stream usage.
     */

    /* Begin insertions */

    insert_wait_batch();
    // A batch of frame is ready

    insert_raw_record();

    if (setting<settings::ComputeMode>() == Computation::Raw)
    {
        insert_dequeue_input();
        return;
    }

    insert_wait_time_stride();

    if (setting<holovibes::settings::DataType>() == RecordedDataType::MOMENTS)
    {
        // Dequeuing the 3 moments in a temporary buffer
        converts_->insert_float_dequeue(input_queue_, moments_env_.moment_tmp_buffer);

        // Splitting them into their respective buffers
        fourier_transforms_->insert_moments_split();

        fourier_transforms_->insert_moments_to_output();
    }
    else
    {
        insert_raw_view();

        converts_->insert_complex_conversion(input_queue_);

        // Spatial transform
        fourier_transforms_->insert_fft(input_queue_.get_fd().width, input_queue_.get_fd().height);

        // Move frames from gpu_space_transformation_buffer to
        // gpu_time_transformation_queue (with respect to
        // time_stride)
        insert_transfer_for_time_transformation();
        insert_wait_time_transformation_size();

        // time transform
        fourier_transforms_->insert_time_transform();
        fourier_transforms_->insert_time_transformation_cuts_view(input_queue_.get_fd(),
                                                                  buffers_.gpu_postprocess_frame_xz.get(),
                                                                  buffers_.gpu_postprocess_frame_yz.get());

        insert_cuts_record();

        // Used for phase increase
        fourier_transforms_->insert_store_p_frame();

        converts_->insert_to_float(buffers_.gpu_postprocess_frame.get());

        insert_moments();
        insert_moments_record();
    }

    insert_filter2d_view();

    // Postprocessing'
    postprocess_->insert_convolution(buffers_.gpu_postprocess_frame.get(), buffers_.gpu_convolution_buffer.get());
    postprocess_->insert_renormalize(buffers_.gpu_postprocess_frame.get());

    // Rendering
    rendering_->insert_fft_shift();

    registration_->insert_registration();

    image_accumulation_->insert_image_accumulation(*buffers_.gpu_postprocess_frame,
                                                   buffers_.gpu_postprocess_frame_size,
                                                   *buffers_.gpu_postprocess_frame_xz,
                                                   *buffers_.gpu_postprocess_frame_yz);

    registration_->set_gpu_reference_image();

    rendering_->insert_chart();
    rendering_->insert_log();
    rendering_->insert_contrast();

    converts_->insert_to_ushort();

    insert_output_enqueue_hologram_mode();

    insert_hologram_record();

    /* The device run asynchronously on its stream (compute stream) with respect
     * to the host The host call device functions, then continues its execution
     * path.
     * We need at some point to synchronize the host with the compute
     * stream.
     * If not, the host will keep on adding new functions to be executed
     * by the device, never letting the device the time to execute them.
     */
    fn_compute_vect_->push_back([&]() { cudaXStreamSynchronize(stream_); });
}

void Pipe::insert_wait_batch()
{
    fn_compute_vect_->push_back(
        [&input_queue_ = input_queue_]()
        {
            // Wait while the input queue is enough filled
            while (input_queue_.is_empty())
                continue;
        });
}

void Pipe::insert_wait_time_stride()
{
    fn_compute_vect_->push_back(
        [this]()
        {
            batch_env_.batch_index += setting<settings::BatchSize>();

            if (batch_env_.batch_index != setting<settings::TimeStride>())
            {
                input_queue_.dequeue();
                fn_compute_vect_->exit_now();
                return;
            }

            batch_env_.batch_index = 0;
        });
}

void Pipe::insert_wait_time_transformation_size()
{
    fn_compute_vect_->push_back(
        [this]()
        {
            if (time_transformation_env_.gpu_time_transformation_queue->get_size() <
                setting<settings::TimeTransformationSize>())
                fn_compute_vect_->exit_now();
        });
}

void Pipe::insert_moments()
{
    bool recording =
        setting<settings::RecordMode>() == RecordMode::MOMENTS && setting<settings::FrameAcquisitionEnabled>();
    ImgType type = setting<settings::ImageType>();

    if (recording || type == ImgType::Moments_0 || type == ImgType::Moments_1 || type == ImgType::Moments_2)
    {
        auto p = setting<settings::P>();
        moments_env_.f_start = p.start;
        moments_env_.f_end =
            std::min<int>(p.start + p.width, static_cast<int>(setting<settings::TimeTransformationSize>()) - 1);

        converts_->insert_to_modulus_moments(moments_env_.stft_res_buffer, moments_env_.f_start, moments_env_.f_end);

        fourier_transforms_->insert_moments();

        if (setting<settings::RegistrationEnabled>())
        {
            registration_->shift_image(moments_env_.moment0_buffer);
            registration_->shift_image(moments_env_.moment1_buffer);
            registration_->shift_image(moments_env_.moment2_buffer);
        }

        fourier_transforms_->insert_moments_to_output();
    }
}

void Pipe::insert_transfer_for_time_transformation()
{
    fn_compute_vect_->push_back(
        [this]()
        {
            time_transformation_env_.gpu_time_transformation_queue->enqueue_multiple(
                buffers_.gpu_spatial_transformation_buffer.get(),
                setting<settings::BatchSize>(),
                stream_);
        });
}

void Pipe::safe_enqueue_output(Queue& output_queue, unsigned short* frame, const std::string& error)
{
    if (!output_queue.enqueue(frame, stream_))
        throw EnqueueException(error);
}

void Pipe::insert_dequeue_input()
{
    fn_compute_vect_->push_back(
        [this]()
        {
            (*processed_output_fps_) += setting<settings::BatchSize>();

            safe_enqueue_output(*buffers_.gpu_output_queue.get(),
                                static_cast<unsigned short*>(input_queue_.get_last_image()),
                                "Can't enqueue the input frame in gpu_output_queue");

            // Dequeue a batch
            input_queue_.dequeue();
        });
}

void Pipe::insert_output_enqueue_hologram_mode()
{
    LOG_FUNC();

    fn_compute_vect_->push_back(
        [this]()
        {
            (*processed_output_fps_)++;

            safe_enqueue_output(*buffers_.gpu_output_queue.get(),
                                buffers_.gpu_output_frame.get(),
                                "Can't enqueue the output frame in gpu_output_queue");

            // Always enqueue the cuts if enabled
            if (setting<settings::CutsViewEnabled>())
            {
                safe_enqueue_output(*time_transformation_env_.gpu_output_queue_xz.get(),
                                    buffers_.gpu_output_frame_xz.get(),
                                    "Can't enqueue the output xz frame in output xz queue");

                safe_enqueue_output(*time_transformation_env_.gpu_output_queue_yz.get(),
                                    buffers_.gpu_output_frame_yz.get(),
                                    "Can't enqueue the output yz frame in output yz queue");
            }

            if (setting<settings::Filter2dViewEnabled>())
            {
                safe_enqueue_output(*gpu_filter2d_view_queue_.get(),
                                    buffers_.gpu_filter2d_frame.get(),
                                    "Can't enqueue the output frame in "
                                    "gpu_filter2d_view_queue");
            }
        });
}

void Pipe::insert_filter2d_view()
{
    if (setting<settings::Filter2dEnabled>() && setting<settings::Filter2dViewEnabled>())
    {
        fn_compute_vect_->push_back(
            [this]()
            {
                int width = buffers_.gpu_output_queue->get_fd().width;
                int height = buffers_.gpu_output_queue->get_fd().height;

                shift_corners(buffers_.gpu_complex_filter2d_frame.get(), 1, width, height, stream_);

                complex_to_modulus(buffers_.gpu_float_filter2d_frame.get(),
                                   buffers_.gpu_complex_filter2d_frame.get(),
                                   0,
                                   0,
                                   buffers_.gpu_postprocess_frame_size,
                                   stream_);
            });
    }
}

void Pipe::insert_raw_view()
{
    if (!setting<settings::RawViewEnabled>() || !gpu_raw_view_queue_)
        return;

    // FIXME: Copy multiple copies a batch of frames
    // The view use get last image which will always the
    // last image of the batch.
    fn_compute_vect_->push_back(
        [this]()
        {
            // Copy a batch of frame from the input queue to the raw view
            // queue
            input_queue_.copy_multiple(*get_raw_view_queue(), cudaMemcpyDeviceToDevice);
        });
}

#pragma region Insert Record

void Pipe::insert_raw_record()
{
    if (!setting<settings::FrameAcquisitionEnabled>() || setting<settings::RecordMode>() != RecordMode::RAW)
        return;

    fn_compute_vect_->push_back(
        [&]()
        {
            if (!can_insert_to_record_queue(setting<settings::BatchSize>()))
                return;

            input_queue_.copy_multiple(record_queue_,
                                       setting<settings::BatchSize>(),
                                       get_memcpy_kind<settings::RecordQueueLocation>());
        });
}

void Pipe::insert_moments_record()
{
    if (!setting<settings::FrameAcquisitionEnabled>() || setting<settings::RecordMode>() != RecordMode::MOMENTS)
        return;

    fn_compute_vect_->push_back(
        [&]()
        {
            if (!can_insert_to_record_queue(3))
                return;

            cudaMemcpyKind kind = get_memcpy_kind<settings::RecordQueueLocation>();

            record_queue_.enqueue(moments_env_.moment0_buffer, stream_, kind);
            record_queue_.enqueue(moments_env_.moment1_buffer, stream_, kind);
            record_queue_.enqueue(moments_env_.moment2_buffer, stream_, kind);
        });
}

void Pipe::insert_hologram_record()
{
    if (!setting<settings::FrameAcquisitionEnabled>() || setting<settings::RecordMode>() != RecordMode::HOLOGRAM)
        return;

    fn_compute_vect_->push_back(
        [this]()
        {
            if (!can_insert_to_record_queue(1))
                return;

            if (buffers_.gpu_output_queue->get_fd().depth == camera::PixelDepth::Bits48) // Complex mode
                record_queue_.enqueue_from_48bit(buffers_.gpu_output_frame.get(),
                                                 stream_,
                                                 get_memcpy_kind<settings::RecordQueueLocation>());
            else
                record_queue_.enqueue(buffers_.gpu_output_frame.get(),
                                      stream_,
                                      get_memcpy_kind<settings::RecordQueueLocation>());
        });
}

void Pipe::insert_cuts_record()
{
    if (!setting<settings::FrameAcquisitionEnabled>())
        return;

    auto recordMode = setting<settings::RecordMode>();
    auto buffer = recordMode == RecordMode::CUTS_XZ   ? buffers_.gpu_output_frame_xz.get()
                  : recordMode == RecordMode::CUTS_YZ ? buffers_.gpu_output_frame_yz.get()
                                                      : nullptr;

    if (!buffer)
        return;

    fn_compute_vect_->push_back(
        [this, &buffer = buffer]()
        {
            if (!can_insert_to_record_queue(1))
                return;

            record_queue_.enqueue(buffer, stream_, get_memcpy_kind<settings::RecordQueueLocation>());
        });
}

#pragma endregion

void Pipe::exec()
{
    onrestart_settings_.apply_updates();

    if (is_requested(ICS::Refresh))
        refresh();

    while (!is_requested(ICS::Termination))
    {
        // Run the entire pipeline of calculation
        run_all();

        if (realtime_settings_.updated())
        {
            apply_realtime_settings();

            image_accumulation_->clear(); // Clear the accumulation queue
            rendering_->request_autocontrast();
        }

        if (is_requested(ICS::Refresh) || pipe_refresh_settings_.updated())
            refresh();
    }
}

std::unique_ptr<Queue>& Pipe::get_lens_queue() { return fourier_transforms_->get_lens_queue(); }

void Pipe::run_all() { fn_compute_vect_->call_all(); }

} // namespace holovibes
