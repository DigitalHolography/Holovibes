#include "pipe.hh"

#include "queue.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "logger.hh"

#include "notifier.hh"

#include "filter2D.cuh"
#include "fft1.cuh"
#include "fft2.cuh"
#include "stft.cuh"
#include "convolution.cuh"
#include "composite.cuh"
#include "tools.cuh"
#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "tools.hh"
#include "contrast_correction.cuh"
#include "enqueue_exception.hh"
#include "pipeline_utils.hh"
#include "holovibes.hh"
#include "cuda_memory.cuh"
#include "global_state_holder.hh"

#include "API.hh"

namespace holovibes
{

void Pipe::keep_contiguous(int nb_elm_to_add) const
{
    while (record_queue_.get_size() + nb_elm_to_add > record_queue_.get_max_size() &&
           // This check prevents being stuck in this loop because record might stop while in this loop
           Holovibes::instance().is_recording())
    {
    }
}

using camera::FrameDescriptor;

Pipe::~Pipe() { GSH::fast_updates_map<FpsType>.remove_entry(FpsType::OUTPUT_FPS); }

bool Pipe::make_requests()
{
    // In order to have a better memory management, free all the ressources that needs to be freed first and allocate
    // the ressources that need to beallocated in second

    bool success_allocation = true;

    /* Free buffers */
    if (disable_convolution_requested_)
    {
        LOG_DEBUG("disable_convolution_requested");

        postprocess_->dispose();
        disable_convolution_requested_ = false;
    }

    if (disable_filter_requested_)
    {
        LOG_DEBUG("disable_filter_requested");

        postprocess_->dispose();
        disable_filter_requested_ = false;
    }

    if (request_disable_lens_view_)
    {
        LOG_DEBUG("request_disable_lens_view");

        fourier_transforms_->get_lens_queue().reset(nullptr);

        request_disable_lens_view_ = false;
    }

    if (disable_raw_view_requested_)
    {
        LOG_DEBUG("disable_raw_view_requested");

        gpu_raw_view_queue_.reset(nullptr);
        api::set_raw_view_enabled(false);
        disable_raw_view_requested_ = false;
    }

    if (disable_filter2d_view_requested_)
    {
        LOG_DEBUG("disable_filter2D_view_requested");

        gpu_filter2d_view_queue_.reset(nullptr);
        api::set_filter2d_view_enabled(false);
        disable_filter2d_view_requested_ = false;
    }

    if (request_delete_time_transformation_cuts_)
    {
        LOG_DEBUG("request_delete_time_transformation_cuts");

        dispose_cuts();
        request_delete_time_transformation_cuts_ = false;
    }

    if (disable_chart_display_requested_)
    {
        LOG_DEBUG("disable_chart_display_requested");

        chart_env_.chart_display_queue_.reset(nullptr);
        api::set_chart_display_enabled(false);
        disable_chart_display_requested_ = false;
    }

    if (disable_chart_record_requested_)
    {
        LOG_DEBUG("disable_chart_record_requested");

        chart_env_.chart_record_queue_.reset(nullptr);
        api::set_chart_record_enabled(false);
        chart_env_.nb_chart_points_to_record_ = 0;
        disable_chart_record_requested_ = false;
    }

    if (disable_frame_record_requested_)
    {
        LOG_DEBUG("disable_frame_record_requested");

        record_queue_.reset(); // we only empty the queue, since it is preallocated and stays allocated
        api::set_frame_record_enabled(false);
        disable_frame_record_requested_ = false;
    }

    image_accumulation_->dispose(); // done only if requested

    /* Allocate buffer */
    if (convolution_requested_)
    {
        LOG_DEBUG("convolution_requested");

        postprocess_->init();
        convolution_requested_ = false;
    }

    if (filter_requested_)
    {
        LOG_DEBUG("filter_requested");

        // TODO
        // fourier_transforms_->init();
        api::enable_filter();
        auto filter = api::get_input_filter();
        fourier_transforms_->update_setting(settings::InputFilter{filter});
        filter_requested_ = false;
    }

    if (output_resize_requested_.load() != std::nullopt)
    {
        LOG_DEBUG("output_resize_requested");

        gpu_output_queue_.resize(output_resize_requested_.load().value(), stream_);
        output_resize_requested_ = std::nullopt;
    }

    // Updating number of images
    if (update_time_transformation_size_requested_)
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
        update_time_transformation_size_requested_ = false;
    }

    if (request_update_time_stride_)
    {
        LOG_DEBUG("request_update_time_stride");

        batch_env_.batch_index = 0;
        request_update_time_stride_ = false;
    }

    if (request_update_batch_size_)
    {
        LOG_DEBUG("request_update_batch_size");

        update_spatial_transformation_parameters();
        input_queue_.resize(setting<settings::BatchSize>());
        request_update_batch_size_ = false;
    }

    if (request_time_transformation_cuts_)
    {
        LOG_DEBUG("request_time_transformation_cuts");

        init_cuts();
        request_time_transformation_cuts_ = false;
    }

    image_accumulation_->init(); // done only if requested

    if (request_clear_img_accu)
    {
        LOG_DEBUG("request_clear_img_accu");

        image_accumulation_->clear();
        request_clear_img_accu = false;
    }

    if (raw_view_requested_)
    {
        LOG_DEBUG("raw_view_requested");

        auto fd = input_queue_.get_fd();
        gpu_raw_view_queue_.reset(new Queue(fd,
                                            setting<settings::OutputBufferSize>(),
                                            QueueType::UNDEFINED,
                                            setting<settings::RawViewQueueLocation>()));
        api::set_raw_view_enabled(true);
        raw_view_requested_ = false;
    }

    if (filter2d_view_requested_)
    {
        LOG_DEBUG("filter2d_view_requested");

        auto fd = gpu_output_queue_.get_fd();
        gpu_filter2d_view_queue_.reset(new Queue(fd, setting<settings::OutputBufferSize>()));
        api::set_filter2d_view_enabled(true);
        filter2d_view_requested_ = false;
    }

    if (chart_display_requested_)
    {
        LOG_DEBUG("chart_display_requested");

        chart_env_.chart_display_queue_.reset(new ConcurrentDeque<ChartPoint>());
        api::set_chart_display_enabled(true);
        chart_display_requested_ = false;
    }

    if (chart_record_requested_.load() != std::nullopt)
    {
        LOG_DEBUG("chart_record_requested");

        chart_env_.chart_record_queue_.reset(new ConcurrentDeque<ChartPoint>());
        api::set_chart_record_enabled(true);
        chart_env_.nb_chart_points_to_record_ = chart_record_requested_.load().value();
        chart_record_requested_ = std::nullopt;
    }

    if (frame_record_requested_)
    {
        LOG_DEBUG("Frame Record Request Processing");
        api::set_frame_record_enabled(true);
        frame_record_requested_ = false;
        LOG_DEBUG("Frame Record Request Processed");
    }

    return success_allocation;
}

void Pipe::refresh()
{
    pipe_refresh_apply_updates();

    refresh_requested_ = false;

    fn_compute_vect_.clear();

    // Aborting if allocation failed
    if (!make_requests())
    {
        refresh_requested_ = false;
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

    insert_wait_frames();
    // A batch of frame is ready

    insert_raw_record();

    if (setting<settings::ComputeMode>() == Computation::Raw)
    {
        insert_dequeue_input();
        return;
    }

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

    update_batch_index();

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // !! BELOW ENQUEUE IN FN COMPUTE VECT MUST BE CONDITIONAL PUSH BACK !!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // time transform
    fourier_transforms_->insert_time_transform();
    fourier_transforms_->insert_time_transformation_cuts_view(input_queue_.get_fd(),
                                                              buffers_.gpu_postprocess_frame_xz.get(),
                                                              buffers_.gpu_postprocess_frame_yz.get());
    insert_cuts_record();

    // Used for phase increase
    fourier_transforms_->insert_store_p_frame();

    converts_->insert_to_float(unwrap_2d_requested_, buffers_.gpu_postprocess_frame.get());

    insert_filter2d_view();

    postprocess_->insert_convolution(buffers_.gpu_postprocess_frame.get(), buffers_.gpu_convolution_buffer.get());
    postprocess_->insert_renormalize(buffers_.gpu_postprocess_frame.get());

    auto insert = [&]()
    {
        image_accumulation_->insert_image_accumulation(*buffers_.gpu_postprocess_frame,
                                                       buffers_.gpu_postprocess_frame_size,
                                                       *buffers_.gpu_postprocess_frame_xz,
                                                       *buffers_.gpu_postprocess_frame_yz);
    };
    insert();
    rendering_->insert_fft_shift();
    rendering_->insert_chart();
    rendering_->insert_log();

    insert_request_autocontrast();
    rendering_->insert_contrast(autocontrast_requested_,
                                autocontrast_slice_xz_requested_,
                                autocontrast_slice_yz_requested_,
                                autocontrast_filter2d_requested_);

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
    fn_compute_vect_.conditional_push_back([&]() { cudaXStreamSynchronize(stream_); });

    // Must be the last inserted function
    insert_reset_batch_index();
}

void Pipe::insert_wait_frames()
{
    fn_compute_vect_.push_back(
        [&]()
        {
            // Wait while the input queue is enough filled
            while (input_queue_.is_empty())
                continue;
        });
}

void Pipe::insert_reset_batch_index()
{
    fn_compute_vect_.conditional_push_back([&]() { batch_env_.batch_index = 0; });
}

void Pipe::insert_transfer_for_time_transformation()
{
    fn_compute_vect_.push_back(
        [&]()
        {
            time_transformation_env_.gpu_time_transformation_queue->enqueue_multiple(
                buffers_.gpu_spatial_transformation_buffer.get(),
                setting<settings::BatchSize>(),
                stream_);
        });
}

void Pipe::update_batch_index()
{
    fn_compute_vect_.push_back(
        [&]()
        {
            batch_env_.batch_index += setting<settings::BatchSize>();
            CHECK(batch_env_.batch_index <= setting<settings::TimeStride>(),
                  "batch_index = {}",
                  batch_env_.batch_index);
        });
}

void Pipe::safe_enqueue_output(Queue& output_queue, unsigned short* frame, const std::string& error)
{
    if (!output_queue.enqueue(frame, stream_))
        throw EnqueueException(error);
}

void Pipe::insert_dequeue_input()
{
    fn_compute_vect_.push_back(
        [&]()
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

    fn_compute_vect_.conditional_push_back(
        [&]()
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
        fn_compute_vect_.conditional_push_back(
            [&]()
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
    if (setting<settings::RawViewEnabled>())
    {
        // FIXME: Copy multiple copies a batch of frames
        // The view use get last image which will always the
        // last image of the batch.
        fn_compute_vect_.push_back(
            [&]()
            {
                // Copy a batch of frame from the input queue to the raw view
                // queue
                cudaMemcpyKind memcpy_kind;
                if (setting<settings::InputQueueLocation>() == Device::GPU)
                    memcpy_kind = setting<settings::RawViewQueueLocation>() == Device::GPU ? cudaMemcpyDeviceToDevice
                                                                                           : cudaMemcpyDeviceToHost;
                else
                    memcpy_kind = setting<settings::RawViewQueueLocation>() == Device::GPU ? cudaMemcpyHostToDevice
                                                                                           : cudaMemcpyHostToHost;

                input_queue_.copy_multiple(*get_raw_view_queue(), memcpy_kind);
            });
    }
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
        fn_compute_vect_.push_back([&]() { keep_contiguous(setting<settings::BatchSize>()); });

        fn_compute_vect_.push_back(
            [&]()
            {
                // If the number of frames to record is reached, stop
                if (setting<settings::RecordFrameCount>() != std::nullopt &&
                    inserted >= setting<settings::RecordFrameCount>().value())
                {
                    auto& manager = NotifierManager::get_instance();
                    auto notifier = manager.get_notifier<bool>("acquisition_finished");
                    notifier->notify(true);
                    return;
                }
                cudaMemcpyKind memcpy_kind;
                if (setting<settings::InputQueueLocation>() == Device::GPU)
                    memcpy_kind = setting<settings::RecordQueueLocation>() == Device::GPU ? cudaMemcpyDeviceToDevice
                                                                                          : cudaMemcpyDeviceToHost;
                else
                    memcpy_kind = setting<settings::RecordQueueLocation>() == Device::GPU ? cudaMemcpyHostToDevice
                                                                                          : cudaMemcpyDeviceToHost;

                input_queue_.copy_multiple(record_queue_, setting<settings::BatchSize>(), memcpy_kind);

                inserted += setting<settings::BatchSize>();
            });
    }
}

void Pipe::insert_hologram_record()
{
    if (setting<settings::FrameRecordEnabled>() && setting<settings::RecordMode>() == RecordMode::HOLOGRAM)
    {
        // if (Holovibes::instance().is_cli)
        fn_compute_vect_.push_back([&]() { keep_contiguous(1); });

        fn_compute_vect_.conditional_push_back(
            [&]()
            {
                if (gpu_output_queue_.get_fd().depth == 6) // Complex mode
                    record_queue_.enqueue_from_48bit(buffers_.gpu_output_frame.get(),
                                                     stream_,
                                                     setting<settings::RecordQueueLocation>() == Device::GPU
                                                         ? cudaMemcpyDeviceToDevice
                                                         : cudaMemcpyDeviceToHost);
                else
                    record_queue_.enqueue(buffers_.gpu_output_frame.get(),
                                          stream_,
                                          setting<settings::RecordQueueLocation>() == Device::GPU
                                              ? cudaMemcpyDeviceToDevice
                                              : cudaMemcpyDeviceToHost);
            });
    }
}

void Pipe::insert_cuts_record()
{
    if (setting<settings::FrameRecordEnabled>())
    {
        if (setting<settings::RecordMode>() == RecordMode::CUTS_XZ)
        {
            fn_compute_vect_.push_back(
                [&]()
                {
                    record_queue_.enqueue(buffers_.gpu_output_frame_xz.get(),
                                          stream_,
                                          setting<settings::RecordQueueLocation>() == Device::GPU
                                              ? cudaMemcpyDeviceToDevice
                                              : cudaMemcpyDeviceToHost);
                });
        }
        else if (setting<settings::RecordMode>() == RecordMode::CUTS_YZ)
        {
            fn_compute_vect_.push_back(
                [&]()
                {
                    record_queue_.enqueue(buffers_.gpu_output_frame_yz.get(),
                                          stream_,
                                          setting<settings::RecordQueueLocation>() == Device::GPU
                                              ? cudaMemcpyDeviceToDevice
                                              : cudaMemcpyDeviceToHost);
                });
        }
    }
}

void Pipe::insert_request_autocontrast()
{
    if (api::get_contrast_enabled() && api::get_contrast_auto_refresh())
        request_autocontrast(setting<settings::CurrentWindow>());
}

void Pipe::exec()
{
    onrestart_settings_.apply_updates();

    if (refresh_requested_ && refresh_enabled_)
        refresh();

    while (!termination_requested_)
    {
        try
        {
            // Run the entire pipeline of calculation
            run_all();

            if (refresh_requested_ && refresh_enabled_)
            {
                refresh();
            }
        }
        catch (CustomException& e)
        {
            LOG_ERROR("Pipe error: message: {}", e.what());
            throw;
        }
    }
}

std::unique_ptr<Queue>& Pipe::get_lens_queue() { return fourier_transforms_->get_lens_queue(); }

void Pipe::insert_fn_end_vect(std::function<void()> function)
{
    std::lock_guard<std::mutex> lock(fn_end_vect_mutex_);
    fn_end_vect_.push_back(function);
}

void Pipe::run_all()
{
    for (FnType& f : fn_compute_vect_)
        f();
    {
        std::lock_guard<std::mutex> lock(fn_end_vect_mutex_);
        for (FnType& f : fn_end_vect_)
            f();
        fn_end_vect_.clear();
    }
}

} // namespace holovibes
