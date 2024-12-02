#include "pipe.hh"

#include "queue.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "logger.hh"

#include "notifier.hh"

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

void Pipe::keep_contiguous(int nb_elm_to_add) const
{
    while (record_queue_.get_size() + nb_elm_to_add > record_queue_.get_max_size() &&
           // This check prevents being stuck in this loop because record might stop while in this loop
           api::is_recording())
        continue;
}

using camera::FrameDescriptor;

Pipe::~Pipe() { FastUpdatesMap::map<IntType>.remove_entry(IntType::OUTPUT_FPS); }

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

    /* Free buffers */
    HANDLE_REQUEST(ICS::DisableConvolution, "Disable convolution", postprocess_->dispose());
    HANDLE_REQUEST(ICS::DisableFilter, "Disable filter", postprocess_->dispose());

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
        api::set_filter2d_view_enabled(false);
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
        api::set_chart_display_enabled(false);
        clear_request(ICS::DisableChartDisplay);
    }

    if (is_requested(ICS::DisableChartRecord))
    {
        LOG_DEBUG("disable_chart_record_requested");

        chart_env_.chart_record_queue_.reset(nullptr);
        api::set_chart_record_enabled(false);
        chart_env_.nb_chart_points_to_record_ = 0;
        clear_request(ICS::DisableChartRecord);
    }

    if (is_requested(ICS::DisableFrameRecord))
    {
        LOG_DEBUG("disable_frame_record_requested");

        record_queue_.reset(); // we only empty the queue, since it is preallocated and stays allocated
        api::set_frame_record_enabled(false);
        clear_request(ICS::DisableFrameRecord);
    }

    image_accumulation_->dispose(); // done only if requested

    /* Allocate buffer */
    HANDLE_REQUEST(ICS::UpdateTimeTransformationAlgorithm,
                   "Update time tr. algorithm",
                   perform_time_transformation_setting_specific_tasks(setting<settings::TimeTransformationSize>()));

    HANDLE_REQUEST(ICS::Convolution, "Convolution", postprocess_->init());

    if (is_requested(ICS::Filter))
    {
        LOG_DEBUG("filter_requested");

        // TODO
        // fourier_transforms_->init();
        api::enable_filter();
        auto filter = api::get_input_filter();
        fourier_transforms_->update_setting(settings::InputFilter{filter});
        clear_request(ICS::Filter);
    }

    // Updating number of images
    if (is_requested(ICS::UpdateTimeTransformationSize))
    {
        LOG_DEBUG("update_time_transformation_size_requested");

        if (!update_time_transformation_size(setting<settings::TimeTransformationSize>()))
        {
            success_allocation = false;
            auto P = setting<settings::P>();
            P.start = 0;
            realtime_settings_.update_setting(settings::P{P});
            api::set_time_transformation_size(1);
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

        auto fd = gpu_output_queue_.get_fd();
        gpu_filter2d_view_queue_.reset(new Queue(fd, static_cast<unsigned int>(setting<settings::OutputBufferSize>())));
        api::set_filter2d_view_enabled(true);
        clear_request(ICS::Filter2DView);
    }

    if (is_requested(ICS::ChartDisplay))
    {
        LOG_DEBUG("chart_display_requested");

        chart_env_.chart_display_queue_.reset(new ConcurrentDeque<ChartPoint>());
        api::set_chart_display_enabled(true);
        clear_request(ICS::ChartDisplay);
    }

    if (chart_record_requested_.load() != std::nullopt)
    {
        LOG_DEBUG("chart_record_requested");

        chart_env_.chart_record_queue_.reset(new ConcurrentDeque<ChartPoint>());
        api::set_chart_record_enabled(true);
        chart_env_.nb_chart_points_to_record_ = chart_record_requested_.load().value();
        chart_record_requested_ = std::nullopt;
    }

    if (is_requested(ICS::UpdateRegistrationZone))
    {
        registration_->updade_cirular_mask();
        clear_request(ICS::UpdateRegistrationZone);
    }

    HANDLE_REQUEST(ICS::FrameRecord, "Frame Record", api::set_frame_record_enabled(true));

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

    if (api::get_data_type() == RecordedDataType::MOMENTS)
    {
        // Dequeuing the 3 moments in a row
        converts_->insert_float_dequeue(input_queue_, moments_env_.moment0_buffer);

        converts_->insert_float_dequeue(input_queue_, moments_env_.moment1_buffer);

        converts_->insert_float_dequeue(input_queue_, moments_env_.moment2_buffer);

        fourier_transforms_->insert_moments_to_output();
    }
    else
    {
        insert_raw_view();

        converts_->insert_complex_conversion(input_queue_);

        // Spatial transform
        fourier_transforms_->insert_fft(buffers_.gpu_filter2d_mask.get(),
                                        input_queue_.get_fd().width,
                                        input_queue_.get_fd().height);

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

        converts_->insert_to_float(is_requested(ICS::Unwrap2D), buffers_.gpu_postprocess_frame.get());

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

    // converts_->insert_cuts_final();

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
    bool recording = setting<settings::RecordMode>() == RecordMode::MOMENTS && setting<settings::FrameRecordEnabled>();
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

            // FIXME: It seems this enqueue is useless because the RawWindow use
            // the gpu input queue for display
            /* safe_enqueue_output(
            **    gpu_output_queue_,
            **    static_cast<unsigned short*>(input_queue_.get_start()),
            **    "Can't enqueue the input frame in gpu_output_queue");
            */

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

            safe_enqueue_output(gpu_output_queue_,
                                buffers_.gpu_output_frame.get(),
                                "Can't enqueue the output frame in gpu_output_queue");

            // Always enqueue the cuts if enabled
            if (api::get_cuts_view_enabled())
            {
                safe_enqueue_output(*time_transformation_env_.gpu_output_queue_xz.get(),
                                    buffers_.gpu_output_frame_xz.get(),
                                    "Can't enqueue the output xz frame in output xz queue");

                safe_enqueue_output(*time_transformation_env_.gpu_output_queue_yz.get(),
                                    buffers_.gpu_output_frame_yz.get(),
                                    "Can't enqueue the output yz frame in output yz queue");
            }

            if (api::get_filter2d_view_enabled())
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
    if (api::get_filter2d_enabled() && api::get_filter2d_view_enabled())
    {
        fn_compute_vect_->push_back(
            [this]()
            {
                int width = gpu_output_queue_.get_fd().width;
                int height = gpu_output_queue_.get_fd().height;

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

void Pipe::insert_raw_record()
{

    // Increment the number of frames inserted in the record queue, so that when it bypasses the requested number, the
    // record finishes This counter happens during the enqueing instead of the dequeuing, because the frequency of the
    // input_queue is usually way faster than the gpu_frame_record queue's, and it would cause the overwritting of
    // the record queue When a new record is started, a refresh of the pipe is requested, and this variable is reset
    static size_t inserted = 0;
    inserted = 0;
    if (setting<settings::FrameRecordEnabled>() && setting<settings::RecordMode>() == RecordMode::RAW)
    {
        // if (Holovibes::instance().is_cli)
        fn_compute_vect_->push_back([&]() { keep_contiguous(setting<settings::BatchSize>()); });

        fn_compute_vect_->push_back(
            [&]()
            {
                // If the number of frames to record is reached, stop
                if (setting<settings::RecordFrameCount>() != std::nullopt &&
                    inserted >= setting<settings::RecordFrameCount>().value())
                {
                    NotifierManager::notify<bool>("acquisition_finished", true);
                    return;
                }

                input_queue_.copy_multiple(record_queue_,
                                           setting<settings::BatchSize>(),
                                           get_memcpy_kind<settings::RecordQueueLocation>());

                inserted += setting<settings::BatchSize>();
            });
    }
}

void Pipe::insert_moments_record()
{
    if (setting<settings::FrameRecordEnabled>() && setting<settings::RecordMode>() == RecordMode::MOMENTS)
    {
        // if (Holovibes::instance().is_cli)
        fn_compute_vect_->push_back([&]() { keep_contiguous(3); });

        fn_compute_vect_->push_back(
            [&]()
            {
                cudaMemcpyKind kind = get_memcpy_kind<settings::RecordQueueLocation>();

                record_queue_.enqueue(moments_env_.moment0_buffer, stream_, kind);
                record_queue_.enqueue(moments_env_.moment1_buffer, stream_, kind);
                record_queue_.enqueue(moments_env_.moment2_buffer, stream_, kind);
            });
    }
}

void Pipe::insert_hologram_record()
{
    if (setting<settings::FrameRecordEnabled>() && setting<settings::RecordMode>() == RecordMode::HOLOGRAM)
    {
        // if (Holovibes::instance().is_cli)
        fn_compute_vect_->push_back([this]() { keep_contiguous(1); });

        fn_compute_vect_->push_back(
            [this]()
            {
                if (gpu_output_queue_.get_fd().depth == camera::PixelDepth::Bits48) // Complex mode
                    record_queue_.enqueue_from_48bit(buffers_.gpu_output_frame.get(),
                                                     stream_,
                                                     get_memcpy_kind<settings::RecordQueueLocation>());
                else
                    record_queue_.enqueue(buffers_.gpu_output_frame.get(),
                                          stream_,
                                          get_memcpy_kind<settings::RecordQueueLocation>());
            });
    }
}

void Pipe::insert_cuts_record()
{
    if (!setting<settings::FrameRecordEnabled>())
        return;

    auto recordMode = setting<settings::RecordMode>();
    auto buffer = recordMode == RecordMode::CUTS_XZ   ? buffers_.gpu_output_frame_xz.get()
                  : recordMode == RecordMode::CUTS_YZ ? buffers_.gpu_output_frame_yz.get()
                                                      : nullptr;

    if (buffer != nullptr)
    {
        fn_compute_vect_->push_back(
            [this, &buffer = buffer]()
            { record_queue_.enqueue(buffer, stream_, get_memcpy_kind<settings::RecordQueueLocation>()); });
    }
}

void Pipe::exec()
{
    onrestart_settings_.apply_updates();

    if (is_requested(ICS::Refresh) && is_requested(ICS::RefreshEnabled))
        refresh();

    while (!is_requested(ICS::Termination))
    {
        try
        {
            // Run the entire pipeline of calculation
            run_all();

            if (is_requested(ICS::Refresh) && is_requested(ICS::RefreshEnabled))
                refresh();
        }
        catch (CustomException& e)
        {
            LOG_ERROR("Pipe error: message: {}", e.what());
            throw;
        }
    }
}

std::unique_ptr<Queue>& Pipe::get_lens_queue() { return fourier_transforms_->get_lens_queue(); }

void Pipe::run_all() { fn_compute_vect_->call_all(); }

} // namespace holovibes
