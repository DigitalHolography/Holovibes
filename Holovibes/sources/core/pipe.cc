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

#include "all_pipe_requests_on_sync_functions.hh"

#include "API.hh"

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

Pipe::Pipe(BatchInputQueue& input, Queue& output, const cudaStream_t& stream)
    : ICompute(input, output, stream)
    , processed_output_fps_(GSH::fast_updates_map<FpsType>.create_entry(FpsType::OUTPUT_FPS))
{

    ConditionType batch_condition = [&]() -> bool
    { return batch_env_.batch_index == compute_cache_.get_value<TimeStride>(); };

    fn_compute_vect_ = FunctionVector(batch_condition);
    fn_end_vect_ = FunctionVector(batch_condition);

    image_accumulation_ = std::make_unique<compute::ImageAccumulation>(fn_compute_vect_,
                                                                       image_acc_env_,
                                                                       buffers_,
                                                                       input.get_fd(),
                                                                       stream_,
                                                                       view_cache_);
    fourier_transforms_ = std::make_unique<compute::FourierTransform>(fn_compute_vect_,
                                                                      buffers_,
                                                                      input.get_fd(),
                                                                      spatial_transformation_plan_,
                                                                      time_transformation_env_,
                                                                      stream_,
                                                                      advanced_cache_,
                                                                      compute_cache_,
                                                                      view_cache_);
    rendering_ = std::make_unique<compute::Rendering>(fn_compute_vect_,
                                                      buffers_,
                                                      chart_env_,
                                                      image_acc_env_,
                                                      time_transformation_env_,
                                                      input.get_fd(),
                                                      output.get_fd(),
                                                      stream_,
                                                      advanced_cache_,
                                                      compute_cache_,
                                                      export_cache_,
                                                      view_cache_,
                                                      zone_cache_);
    converts_ = std::make_unique<compute::Converts>(fn_compute_vect_,
                                                    buffers_,
                                                    time_transformation_env_,
                                                    plan_unwrap_2d_,
                                                    input.get_fd(),
                                                    stream_,
                                                    compute_cache_,
                                                    composite_cache_,
                                                    view_cache_,
                                                    zone_cache_);
    postprocess_ = std::make_unique<compute::Postprocessing>(fn_compute_vect_,
                                                             buffers_,
                                                             input.get_fd(),
                                                             stream_,
                                                             advanced_cache_,
                                                             compute_cache_,
                                                             view_cache_);
    *processed_output_fps_ = 0;

    GSH::instance().change_value<TimeTransformationSize>();

    try
    {
        refresh();
    }
    catch (const holovibes::CustomException& e)
    {
        // If refresh fails the compute descriptor settings will be
        // changed to something that should make refresh work
        // (ex: lowering the GPU memory usage)
        LOG_WARN(compute_worker, "Pipe refresh failed, trying one more time with updated compute descriptor");
        LOG_WARN(compute_worker, "Exception: {}", e.what());
        try
        {
            refresh();
        }
        catch (const holovibes::CustomException& e)
        {
            // If it still didn't work holovibes is probably going to freeze
            // and the only thing you can do is restart it manually
            LOG_ERROR(compute_worker, "Pipe could not be initialized, You might want to restart holovibes");
            LOG_ERROR(compute_worker, "Exception: {}", e.what());
            throw e;
        }
    }
}

Pipe::~Pipe() { GSH::fast_updates_map<FpsType>.remove_entry(FpsType::OUTPUT_FPS); }

void Pipe::synchronize_caches_and_make_requests()
{
    PipeRequestOnSync::begin_requests();
    advanced_cache_.synchronize<AdvancedPipeRequestOnSync>(*this);
    compute_cache_.synchronize<ComputePipeRequestOnSync>(*this);
    import_cache_.synchronize<ImportPipeRequestOnSync>(*this);
    export_cache_.synchronize<ExportPipeRequestOnSync>(*this);
    composite_cache_.synchronize<CompositePipeRequestOnSync>(*this);
    view_cache_.synchronize<ViewPipeRequestOnSync>(*this);
    zone_cache_.synchronize<DefaultPipeRequestOnSync>(*this);

    if (PipeRequestOnSync::has_requests_fail())
    {
        LOG_ERROR(main, "Failure when making requests after all caches synchronizations");
        // FIXME : handle pipe requests on sync failure
        return;
    }
}

bool Pipe::caches_has_change_requested()
{
    return advanced_cache_.has_change_requested() || compute_cache_.has_change_requested() ||
           export_cache_.has_change_requested() || import_cache_.has_change_requested() ||
           view_cache_.has_change_requested() || zone_cache_.has_change_requested() ||
           composite_cache_.has_change_requested();
}

void Pipe::refresh()
{
    // LOG_FUNC(main);

    if (!caches_has_change_requested())
    {
        // LOG_TRACE(main, "Pipe refresh doesn't need refresh : caches already syncs");
        return;
    }

    LOG_TRACE(main, "Pipe refresh : Call caches ...");
    synchronize_caches_and_make_requests();

    if (api::get_import_type() == ImportTypeEnum::None)
    {
        LOG_DEBUG(main, "Pipe refresh doesn't need refresh : no import set");
        return;
    }

    if (!PipeRequestOnSync::do_need_pipe_refresh())
    {
        LOG_DEBUG(
            main,
            "Pipe refresh doesn't need refresh : the cache refresh havn't make change that require a pipe refresh");
        return;
    }

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

    fn_compute_vect_.clear();

    /* Begin insertions */

    // Wait for a batch of frame to be ready
    insert_wait_frames();

    insert_raw_record();

    if (compute_cache_.get_value<ComputeMode>() == Computation::Raw)
    {
        insert_dequeue_input();
        return;
    }

    insert_raw_view();

    converts_->insert_complex_conversion(gpu_input_queue_);

    // Spatial transform
    fourier_transforms_->insert_fft();

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
    fourier_transforms_->insert_time_transformation_cuts_view();
    insert_cuts_record();

    // Used for phase increase
    fourier_transforms_->insert_store_p_frame();

    converts_->insert_to_float(GSH::instance().get_value<Unwrap2DRequested>());

    insert_filter2d_view();

    postprocess_->insert_convolution();
    postprocess_->insert_renormalize();

    image_accumulation_->insert_image_accumulation();

    rendering_->insert_fft_shift();
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
    fn_compute_vect_.conditional_push_back([&]() { cudaXStreamSynchronize(stream_); });

    // Must be the last inserted function
    insert_reset_batch_index();
}

void Pipe::run_all()
{
    refresh();

    for (FnType& f : fn_compute_vect_)
        f();
    {
        std::lock_guard<std::mutex> lock(fn_end_vect_mutex_);
        for (FnType& f : fn_end_vect_)
            f();
        fn_end_vect_.clear();
    }
}

void Pipe::exec()
{
    while (!termination_requested_)
    {
        try
        {
            run_all();
        }
        catch (CustomException& e)
        {
            LOG_ERROR(compute_worker, "Pipe error: message: {}", e.what());
            throw;
        }
    }
}

// Insert functions

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
                compute_cache_.get_value<BatchSize>(),
                stream_);
        });
}

void Pipe::update_batch_index()
{
    fn_compute_vect_.push_back(
        [&]()
        {
            batch_env_.batch_index += compute_cache_.get_value<BatchSize>();
            CHECK(batch_env_.batch_index <= compute_cache_.get_value<TimeStride>(),
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
            *processed_output_fps_ += compute_cache_.get_value<BatchSize>();

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
            if (view_cache_.get_value<CutsViewEnabled>())
            {
                safe_enqueue_output(*time_transformation_env_.gpu_output_queue_xz.get(),
                                    buffers_.gpu_output_frame_xz.get(),
                                    "Can't enqueue the output xz frame in output xz queue");

                safe_enqueue_output(*time_transformation_env_.gpu_output_queue_yz.get(),
                                    buffers_.gpu_output_frame_yz.get(),
                                    "Can't enqueue the output yz frame in output yz queue");
            }

            if (view_cache_.get_value<Filter2DViewEnabled>())
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
    if (compute_cache_.get_value<Filter2D>().enabled && view_cache_.get_value<Filter2DViewEnabled>())
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
    if (view_cache_.get_value<RawViewEnabled>())
    {
        // FIXME: Copy multiple copies a batch of frames
        // The view use get last image which will always the
        // last image of the batch.
        fn_compute_vect_.push_back(
            [&]()
            {
                // Copy a batch of frame from the input queue to the raw view
                // queue
                gpu_input_queue_.copy_multiple(*get_raw_view_queue_ptr());
            });
    }
}

void Pipe::insert_raw_record()
{
    if (GSH::instance().get_value<FrameRecordMode>().get_record_mode() == RecordMode::RAW)
    {
        if (Holovibes::instance().is_cli)
            fn_compute_vect_.push_back([&]() { keep_contiguous(compute_cache_.get_value<BatchSize>()); });

        fn_compute_vect_.push_back(
            [&]() {
                gpu_input_queue_.copy_multiple(*frame_record_env_.gpu_frame_record_queue_,
                                               compute_cache_.get_value<BatchSize>());
            });
    }
}

void Pipe::insert_hologram_record()
{
    if (GSH::instance().get_value<FrameRecordMode>().get_record_mode() == RecordMode::HOLOGRAM)
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
    if (GSH::instance().get_value<FrameRecordMode>().is_enable())
    {
        if (GSH::instance().get_value<FrameRecordMode>().get_record_mode() == RecordMode::CUTS_XZ)
        {
            fn_compute_vect_.push_back(
                [&]()
                { frame_record_env_.gpu_frame_record_queue_->enqueue(buffers_.gpu_output_frame_xz.get(), stream_); });
        }
        else if (GSH::instance().get_value<FrameRecordMode>().get_record_mode() == RecordMode::CUTS_YZ)
        {
            fn_compute_vect_.push_back(
                [&]()
                { frame_record_env_.gpu_frame_record_queue_->enqueue(buffers_.gpu_output_frame_yz.get(), stream_); });
        }
    }
}

void Pipe::insert_fn_end_vect(std::function<void()> function)
{
    std::lock_guard<std::mutex> lock(fn_end_vect_mutex_);
    fn_end_vect_.push_back(function);
}

} // namespace holovibes
