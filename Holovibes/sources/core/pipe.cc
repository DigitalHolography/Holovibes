#include "pipe.hh"

#include "queue.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "logger.hh"

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

namespace holovibes
{

void Pipe::keep_contiguous(int nb_elm_to_add) const
{
    while (frame_record_env_.gpu_frame_record_queue_->get_size() + nb_elm_to_add >
               frame_record_env_.gpu_frame_record_queue_->get_max_size() &&
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
        GSH::instance().set_raw_view_enabled(false);
        disable_raw_view_requested_ = false;
    }

    if (disable_filter2d_view_requested_)
    {
        LOG_DEBUG("disable_filter2D_view_requested");

        gpu_filter2d_view_queue_.reset(nullptr);
        GSH::instance().set_filter2d_view_enabled(false);
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
        GSH::instance().set_chart_display_enabled(false);
        disable_chart_display_requested_ = false;
    }

    if (disable_chart_record_requested_)
    {
        LOG_DEBUG("disable_chart_record_requested");

        chart_env_.chart_record_queue_.reset(nullptr);
        GSH::instance().set_chart_record_enabled(false);
        chart_env_.nb_chart_points_to_record_ = 0;
        disable_chart_record_requested_ = false;
    }

    if (disable_frame_record_requested_)
    {
        LOG_DEBUG("disable_frame_record_requested");

        frame_record_env_.gpu_frame_record_queue_.reset(nullptr);
        frame_record_env_.record_mode_ = RecordMode::NONE;
        GSH::instance().set_frame_record_enabled(false);
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

        if (!update_time_transformation_size(compute_cache_.get_time_transformation_size()))
        {
            success_allocation = false;
            GSH::instance().set_p_index(0);
            GSH::instance().set_time_transformation_size(1);
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
        gpu_input_queue_.resize(compute_cache_.get_batch_size());
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

        auto fd = gpu_input_queue_.get_fd();
        gpu_raw_view_queue_.reset(new Queue(fd, GSH::instance().get_output_buffer_size()));
        GSH::instance().set_raw_view_enabled(true);
        raw_view_requested_ = false;
    }

    if (filter2d_view_requested_)
    {
        LOG_DEBUG("filter2d_view_requested");

        auto fd = gpu_output_queue_.get_fd();
        gpu_filter2d_view_queue_.reset(new Queue(fd, GSH::instance().get_output_buffer_size()));
        GSH::instance().set_filter2d_view_enabled(true);
        filter2d_view_requested_ = false;
    }

    if (chart_display_requested_)
    {
        LOG_DEBUG("chart_display_requested");

        chart_env_.chart_display_queue_.reset(new ConcurrentDeque<ChartPoint>());
        GSH::instance().set_chart_display_enabled(true);
        chart_display_requested_ = false;
    }

    if (chart_record_requested_.load() != std::nullopt)
    {
        LOG_DEBUG("chart_record_requested");

        chart_env_.chart_record_queue_.reset(new ConcurrentDeque<ChartPoint>());
        GSH::instance().set_chart_record_enabled(true);
        chart_env_.nb_chart_points_to_record_ = chart_record_requested_.load().value();
        chart_record_requested_ = std::nullopt;
    }

    if (hologram_record_requested_)
    {
        LOG_DEBUG("Hologram Record Request Processing");
        auto record_fd = gpu_output_queue_.get_fd();
        record_fd.depth = record_fd.depth == 6 ? 3 : record_fd.depth;
        frame_record_env_.gpu_frame_record_queue_.reset(
            new Queue(record_fd, GSH::instance().get_record_buffer_size(), QueueType::RECORD_QUEUE));
        GSH::instance().set_frame_record_enabled(true);
        frame_record_env_.record_mode_ = RecordMode::HOLOGRAM;
        hologram_record_requested_ = false;
        LOG_DEBUG("Hologram Record Request Processed");
    }

    if (raw_record_requested_)
    {
        LOG_DEBUG("Raw Record Request Processing");
        frame_record_env_.gpu_frame_record_queue_.reset(
            new Queue(gpu_input_queue_.get_fd(), GSH::instance().get_record_buffer_size(), QueueType::RECORD_QUEUE));

        GSH::instance().set_frame_record_enabled(true);
        frame_record_env_.record_mode_ = RecordMode::RAW;
        raw_record_requested_ = false;
        LOG_DEBUG("Raw Record Request Processed");
    }

    if (cuts_record_requested_)
    {
        LOG_DEBUG("cuts_record_requested");

        camera::FrameDescriptor fd_xyz = gpu_output_queue_.get_fd();

        fd_xyz.depth = sizeof(ushort);
        if (frame_record_env_.record_mode_ == RecordMode::CUTS_XZ)
            fd_xyz.height = compute_cache_.get_time_transformation_size();
        else
            fd_xyz.width = compute_cache_.get_time_transformation_size();

        frame_record_env_.gpu_frame_record_queue_.reset(
            new Queue(fd_xyz, GSH::instance().get_record_buffer_size(), QueueType::RECORD_QUEUE));

        GSH::instance().set_frame_record_enabled(true);
        cuts_record_requested_ = false;
    }

    return success_allocation;
}

void Pipe::refresh()
{
    // This call has to be before make_requests() because this method needs
    // to get updated values during exec_all() call
    // This call could be removed if make_requests() only gets value through
    // reference caches as such: GSH::instance().get_*() instead of *_cache_.get_*()
    synchronize_caches();

    refresh_requested_ = false;

    fn_compute_vect_.clear();

    // Aborting if allocation failed
    if (!make_requests())
    {
        refresh_requested_ = false;
        return;
    }

    // This call has to be after make_requests() because this method needs
    // to honor cache modifications
    synchronize_caches();

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

    if (compute_cache_.get_compute_mode() == Computation::Raw)
    {
        insert_dequeue_input();
        return;
    }

    insert_raw_view();

    converts_->insert_complex_conversion(gpu_input_queue_);

    // Spatial transform
    fourier_transforms_->insert_fft(buffers_.gpu_filter2d_mask.get(),
                                    gpu_input_queue_.get_fd().width,
                                    gpu_input_queue_.get_fd().height,
                                    filter2d_cache_.get_filter2d_n1(),
                                    filter2d_cache_.get_filter2d_n2(),
                                    filter2d_cache_.get_filter2d_smooth_low(),
                                    filter2d_cache_.get_filter2d_smooth_high(),
                                    compute_cache_.get_space_transformation(),
                                    view_cache_.get_filter2d_enabled());

    // Move frames from gpu_space_transformation_buffer to
    // gpu_time_transformation_queue (with respect to
    // time_stride)
    insert_transfer_for_time_transformation();

    update_batch_index();

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // !! BELOW ENQUEUE IN FN COMPUTE VECT MUST BE CONDITIONAL PUSH BACK !!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // time transform
    fourier_transforms_->insert_time_transform(compute_cache_.get_time_transformation(),
                                               compute_cache_.get_time_transformation_size());
    fourier_transforms_->insert_time_transformation_cuts_view(gpu_input_queue_.get_fd(),
                                                              view_cache_.get_cuts_view_enabled(),
                                                              view_cache_.get_x(),
                                                              view_cache_.get_y(),
                                                              view_cache_.get_xz_const_ref().output_image_accumulation,
                                                              view_cache_.get_yz_const_ref().output_image_accumulation,
                                                              buffers_.gpu_postprocess_frame_xz.get(),
                                                              buffers_.gpu_postprocess_frame_yz.get(),
                                                              view_cache_.get_img_type(),
                                                              compute_cache_.get_time_transformation_size());
    insert_cuts_record();

    // Used for phase increase
    fourier_transforms_->insert_store_p_frame(view_cache_.get_p().start);

    converts_->insert_to_float(unwrap_2d_requested_,
                               view_cache_.get_img_type(),
                               compute_cache_.get_time_transformation(),
                               buffers_.gpu_postprocess_frame.get(),
                               view_cache_.get_p(),
                               compute_cache_.get_time_transformation_size(),
                               composite_cache_.get_rgb(),
                               composite_cache_.get_composite_kind(),
                               composite_cache_.get_composite_auto_weights(),
                               composite_cache_.get_hsv_const_ref(),
                               zone_cache_.get_composite_zone(),
                               compute_cache_.get_unwrap_history_size());

    insert_filter2d_view();

    postprocess_->insert_convolution(compute_cache_.get_convolution_enabled(),
                                     compute_cache_.get_convo_matrix_const_ref(),
                                     view_cache_.get_img_type(),
                                     buffers_.gpu_postprocess_frame.get(),
                                     buffers_.gpu_convolution_buffer.get(),
                                     compute_cache_.get_divide_convolution_enabled());
    postprocess_->insert_renormalize(view_cache_.get_renorm_enabled(),
                                     view_cache_.get_img_type(),
                                     buffers_.gpu_postprocess_frame.get(),
                                     advanced_cache_.get_renorm_constant());

    // !!!!!!!!!!!!!!!!!!!!!!!!!!
    // !! ICI LA LAL ALLALALAL view cache en argumment en bas!!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!
    auto insert = [&]()
    {
        image_accumulation_->insert_image_accumulation(view_cache_.get_xy_const_ref(),
                                                       *buffers_.gpu_postprocess_frame,
                                                       buffers_.gpu_postprocess_frame_size,
                                                       view_cache_.get_xz_const_ref(),
                                                       *buffers_.gpu_postprocess_frame_xz,
                                                       view_cache_.get_yz_const_ref(),
                                                       *buffers_.gpu_postprocess_frame_yz,
                                                       view_cache_);
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
            while (gpu_input_queue_.is_empty())
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
                compute_cache_.get_batch_size(),
                stream_);
        });
}

void Pipe::update_batch_index()
{
    fn_compute_vect_.push_back(
        [&]()
        {
            batch_env_.batch_index += compute_cache_.get_batch_size();
            CHECK(batch_env_.batch_index <= compute_cache_.get_time_stride(),
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
            (*processed_output_fps_) += compute_cache_.get_batch_size();

            // FIXME: It seems this enqueue is useless because the RawWindow use
            // the gpu input queue for display
            /* safe_enqueue_output(
            **    gpu_output_queue_,
            **    static_cast<unsigned short*>(gpu_input_queue_.get_start()),
            **    "Can't enqueue the input frame in gpu_output_queue");
            */

            // Dequeue a batch
            gpu_input_queue_.dequeue();
        });
}

void Pipe::insert_output_enqueue_hologram_mode()
{
    fn_compute_vect_.conditional_push_back(
        [&]()
        {
            (*processed_output_fps_)++;

            safe_enqueue_output(gpu_output_queue_,
                                buffers_.gpu_output_frame.get(),
                                "Can't enqueue the output frame in gpu_output_queue");

            // Always enqueue the cuts if enabled
            if (view_cache_.get_cuts_view_enabled())
            {
                safe_enqueue_output(*time_transformation_env_.gpu_output_queue_xz.get(),
                                    buffers_.gpu_output_frame_xz.get(),
                                    "Can't enqueue the output xz frame in output xz queue");

                safe_enqueue_output(*time_transformation_env_.gpu_output_queue_yz.get(),
                                    buffers_.gpu_output_frame_yz.get(),
                                    "Can't enqueue the output yz frame in output yz queue");
            }

            if (view_cache_.get_filter2d_view_enabled())
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
    if (view_cache_.get_filter2d_enabled() && view_cache_.get_filter2d_view_enabled())
    {
        fn_compute_vect_.conditional_push_back(
            [&]()
            {
                float_to_complex(buffers_.gpu_complex_filter2d_frame.get(),
                                 buffers_.gpu_postprocess_frame.get(),
                                 buffers_.gpu_postprocess_frame_size,
                                 stream_);

                int width = gpu_output_queue_.get_fd().width;
                int height = gpu_output_queue_.get_fd().height;
                CufftHandle handle{width, height, CUFFT_C2C};

                cufftSafeCall(cufftExecC2C(handle,
                                           buffers_.gpu_complex_filter2d_frame.get(),
                                           buffers_.gpu_complex_filter2d_frame.get(),
                                           CUFFT_FORWARD));
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
    if (view_cache_.get_raw_view_enabled())
    {
        // FIXME: Copy multiple copies a batch of frames
        // The view use get last image which will always the
        // last image of the batch.
        fn_compute_vect_.push_back(
            [&]()
            {
                // Copy a batch of frame from the input queue to the raw view
                // queue
                gpu_input_queue_.copy_multiple(*get_raw_view_queue());
            });
    }
}

void Pipe::insert_raw_record()
{
    if (export_cache_.get_frame_record_enabled() && frame_record_env_.record_mode_ == RecordMode::RAW)
    {
        if (Holovibes::instance().is_cli)
            fn_compute_vect_.push_back([&]() { keep_contiguous(compute_cache_.get_batch_size()); });

        fn_compute_vect_.push_back(
            [&]() {
                gpu_input_queue_.copy_multiple(*frame_record_env_.gpu_frame_record_queue_,
                                               compute_cache_.get_batch_size());
            });
    }
}

void Pipe::insert_hologram_record()
{
    if (export_cache_.get_frame_record_enabled() && frame_record_env_.record_mode_ == RecordMode::HOLOGRAM)
    {
        if (Holovibes::instance().is_cli)
            fn_compute_vect_.push_back([&]() { keep_contiguous(1); });

        fn_compute_vect_.conditional_push_back(
            [&]()
            {
                if (gpu_output_queue_.get_fd().depth == 6) // Complex mode
                    frame_record_env_.gpu_frame_record_queue_->enqueue_from_48bit(buffers_.gpu_output_frame.get(),
                                                                                  stream_);
                else
                    frame_record_env_.gpu_frame_record_queue_->enqueue(buffers_.gpu_output_frame.get(), stream_);
            });
    }
}

void Pipe::insert_cuts_record()
{
    if (GSH::instance().get_frame_record_enabled())
    {
        if (frame_record_env_.record_mode_ == RecordMode::CUTS_XZ)
        {
            fn_compute_vect_.push_back(
                [&]()
                { frame_record_env_.gpu_frame_record_queue_->enqueue(buffers_.gpu_output_frame_xz.get(), stream_); });
        }
        else if (frame_record_env_.record_mode_ == RecordMode::CUTS_YZ)
        {
            fn_compute_vect_.push_back(
                [&]()
                { frame_record_env_.gpu_frame_record_queue_->enqueue(buffers_.gpu_output_frame_yz.get(), stream_); });
        }
    }
}

void Pipe::insert_request_autocontrast()
{
    if (GSH::instance().get_contrast_enabled() && GSH::instance().get_contrast_auto_refresh())
        request_autocontrast(view_cache_.get_current_window());
}

void Pipe::exec()
{
    if (refresh_requested_)
        refresh();

    synchronize_caches();

    while (!termination_requested_)
    {
        try
        {
            // Run the entire pipeline of calculation
            run_all();

            if (refresh_requested_)
            {
                refresh();
                synchronize_caches();
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
    synchronize_caches();

    for (FnType& f : fn_compute_vect_)
        f();
    {
        std::lock_guard<std::mutex> lock(fn_end_vect_mutex_);
        for (FnType& f : fn_end_vect_)
            f();
        fn_end_vect_.clear();
    }
}

void Pipe::synchronize_caches()
{
    compute_cache_.synchronize();
    export_cache_.synchronize();
    filter2d_cache_.synchronize();
    view_cache_.synchronize();
    zone_cache_.synchronize();
    composite_cache_.synchronize();
    // never updated during the life time of the app
    // all updated params will be catched on json file when the app will load
    // advanced_cache_.synchronize();
}
} // namespace holovibes
