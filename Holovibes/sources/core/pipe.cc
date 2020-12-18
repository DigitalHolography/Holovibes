/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

#include "pipe.hh"
#include "config.hh"
#include "compute_descriptor.hh"
#include "queue.hh"
#include "compute_bundles.hh"
#include "compute_bundles_2d.hh"
#include "logger.hh"

#include "fft1.cuh"
#include "fft2.cuh"
#include "filter2D.cuh"
#include "stft.cuh"
#include "convolution.cuh"
#include "composite.cuh"
#include "tools.cuh"
#include "tools_conversion.cuh"
#include "tools_compute.cuh"
#include "tools.hh"
#include "contrast_correction.cuh"
#include "custom_exception.hh"
#include "pipeline_utils.hh"
#include "holovibes.hh"
#include "cuda_memory.cuh"

namespace holovibes
{
using camera::FrameDescriptor;

Pipe::Pipe(Queue& input,
           Queue& output,
           ComputeDescriptor& desc,
           const cudaStream_t& stream)
    : ICompute(input, output, desc, stream)
{
    ConditionType batch_condition = [&]() -> bool {
        return batch_env_.batch_index == cd_.time_transformation_stride;
    };

    fn_compute_vect_ = FunctionVector(batch_condition);
    fn_end_vect_ = FunctionVector(batch_condition);

    image_accumulation_ =
        std::make_unique<compute::ImageAccumulation>(fn_compute_vect_,
                                                     image_acc_env_,
                                                     buffers_,
                                                     input.get_fd(),
                                                     desc,
                                                     stream_);
    fourier_transforms_ = std::make_unique<compute::FourierTransform>(
        fn_compute_vect_,
        buffers_,
        input.get_fd(),
        desc,
        spatial_transformation_plan_,
        batch_env_,
        time_transformation_env_,
        stream_);
    rendering_ = std::make_unique<compute::Rendering>(fn_compute_vect_,
                                                      buffers_,
                                                      chart_env_,
                                                      image_acc_env_,
                                                      time_transformation_env_,
                                                      desc,
                                                      input.get_fd(),
                                                      output.get_fd(),
                                                      this,
                                                      stream_);
    converts_ = std::make_unique<compute::Converts>(fn_compute_vect_,
                                                    buffers_,
                                                    batch_env_,
                                                    time_transformation_env_,
                                                    plan_unwrap_2d_,
                                                    desc,
                                                    input.get_fd(),
                                                    output.get_fd(),
                                                    stream_);
    postprocess_ = std::make_unique<compute::Postprocessing>(fn_compute_vect_,
                                                             buffers_,
                                                             input.get_fd(),
                                                             desc,
                                                             stream_);

    update_time_transformation_size_requested_ = true;
    processed_output_fps_.store(0);

    try
    {
        refresh();
    }
    catch (const holovibes::CustomException& e)
    {
        // If refresh() fails the compute descriptor settings will be
        // changed to something that should make refresh() work
        // (ex: lowering the GPU memory usage)
        LOG_WARN("Pipe refresh failed, trying one more time with updated "
                 "compute descriptor");
        try
        {
            refresh();
        }
        catch (const holovibes::CustomException& e)
        {
            // If it still didn't work holovibes is probably going to freeze
            // and the only thing you can do is restart it manually
            LOG_ERROR("Pipe could not be initialized");
            LOG_ERROR("You might want to restart holovibes");
            throw e;
        }
    }
}

bool Pipe::make_requests()
{
    // In order to have a better memory management, free all the ressources
    // that needs to be freed first and allocate the ressources that need
    // to be allocated in second

    bool success_allocation = true;

    /* Free buffers */
    if (disable_convolution_requested_)
    {
        postprocess_->dispose();
        cd_.convolution_enabled = false;
        disable_convolution_requested_ = false;
    }

    if (request_disable_lens_view_)
    {
        fourier_transforms_->get_lens_queue().reset(nullptr);
        request_disable_lens_view_ = false;
    }

    if (disable_raw_view_requested_)
    {
        gpu_raw_view_queue_.reset(nullptr);
        cd_.raw_view_enabled = false;
        disable_raw_view_requested_ = false;
    }

    if (request_delete_time_transformation_cuts_)
    {
        dispose_cuts();
        request_delete_time_transformation_cuts_ = false;
    }

    if (disable_chart_display_requested_)
    {
        chart_env_.chart_display_queue_.reset(nullptr);
        cd_.chart_display_enabled = false;
        disable_chart_display_requested_ = false;
    }

    if (disable_chart_record_requested_)
    {
        chart_env_.chart_record_queue_.reset(nullptr);
        cd_.chart_record_enabled = false;
        chart_env_.nb_chart_points_to_record_ = 0;
        disable_chart_record_requested_ = false;
    }

    if (disable_frame_record_requested_)
    {
        frame_record_env_.gpu_frame_record_queue_.reset(nullptr);
        cd_.frame_record_enabled = false;
        frame_record_env_.remaining_frames_to_record = 0;
        disable_frame_record_requested_ = false;
    }

    image_accumulation_->dispose(); // done only if requested

    /* Allocate buffer */
    if (convolution_requested_)
    {
        postprocess_->init();
        cd_.convolution_enabled = true;
        convolution_requested_ = false;
    }

    if (output_resize_requested_.load() != std::nullopt)
    {
        gpu_output_queue_.resize(output_resize_requested_.load().value(),
                                 stream_);
        output_resize_requested_ = std::nullopt;
    }

    // Updating number of images
    if (update_time_transformation_size_requested_)
    {
        if (!update_time_transformation_size(cd_.time_transformation_size))
        {
            success_allocation = false;
            cd_.pindex = 0;
            cd_.time_transformation_size = 1;
            update_time_transformation_size(1);
            LOG_WARN("Updating #img failed, #img updated to 1");
        }
        update_time_transformation_size_requested_ = false;
    }

    if (request_update_time_transformation_stride_)
    {
        batch_env_.batch_index = 0;
        request_update_time_transformation_stride_ = false;
    }

    if (request_update_batch_size_)
    {
        update_spatial_transformation_parameters();
        request_update_batch_size_ = false;
    }

    if (request_time_transformation_cuts_)
    {
        init_cuts();
        request_time_transformation_cuts_ = false;
    }

    image_accumulation_->init(); // done only if requested

    if (request_clear_img_acc_)
    {
        image_accumulation_->clear();
        request_clear_img_acc_ = false;
    }

    if (raw_view_requested_)
    {
        auto fd = gpu_input_queue_.get_fd();
        gpu_raw_view_queue_.reset(
            new Queue(fd, global::global_config.output_queue_max_size));
        cd_.raw_view_enabled = true;
        raw_view_requested_ = false;
    }

    if (chart_display_requested_)
    {
        chart_env_.chart_display_queue_.reset(
            new ConcurrentDeque<ChartPoint>());
        cd_.chart_display_enabled = true;
        chart_display_requested_ = false;
    }

    if (chart_record_requested_.load() != std::nullopt)
    {
        chart_env_.chart_record_queue_.reset(new ConcurrentDeque<ChartPoint>());
        cd_.chart_record_enabled = true;
        chart_env_.nb_chart_points_to_record_ =
            chart_record_requested_.load().value();
        chart_record_requested_ = std::nullopt;
    }

    if (hologram_record_requested_.load() != std::nullopt)
    {
        auto record_fd = gpu_output_queue_.get_fd();
        record_fd.depth = record_fd.depth == 6 ? 3 : record_fd.depth;
        frame_record_env_.gpu_frame_record_queue_.reset(
            new Queue(record_fd,
                      global::global_config.frame_record_queue_max_size,
                      Queue::QueueType::RECORD_QUEUE));
        cd_.frame_record_enabled = true;
        frame_record_env_.remaining_frames_to_record =
            hologram_record_requested_.load().value();
        frame_record_env_.raw_record_enabled = false;
        hologram_record_requested_ = std::nullopt;
    }

    if (raw_record_requested_.load() != std::nullopt)
    {
        frame_record_env_.gpu_frame_record_queue_.reset(
            new Queue(gpu_input_queue_.get_fd(),
                      global::global_config.frame_record_queue_max_size,
                      Queue::QueueType::RECORD_QUEUE));
        cd_.frame_record_enabled = true;
        frame_record_env_.remaining_frames_to_record =
            raw_record_requested_.load().value();
        frame_record_env_.raw_record_enabled = true;
        raw_record_requested_ = std::nullopt;
    }

    return success_allocation;
}

void Pipe::refresh()
{
    refresh_requested_ = false;

    fn_compute_vect_.clear();

    std::cout << "@make_resquest before" << std::endl;

    // Aborting if allocation failed
    if (!make_requests())
    {
        refresh_requested_ = false;
        return;
    }

    std::cout << "@make_resquest after" << std::endl;

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
     * thread stream with all other streams (reducing performences drastically
     * because of a longer waiting time).
     *
     * The value of the default stream is 0. However, not providing a stream to
     * cuda calls forces the default stream usage.
     */

    /* Begin insertions */

    if (cd_.compute_mode == Computation::Raw)
    {
        refresh_requested_ = false;
        insert_raw_record();
        insert_output_enqueue_raw_mode();
        return;
    }

    const camera::FrameDescriptor& input_fd = gpu_input_queue_.get_fd();

    std::cout << "@insert_wait_frames before" << std::endl;

    insert_wait_frames();

    std::cout << "@insert_wait_frames after" << std::endl;

    insert_raw_record();

    insert_raw_view();

    converts_->insert_complex_conversion(gpu_input_queue_);

    std::cout << "@insert_fft before" << std::endl;

    // Spatial transform
    fourier_transforms_->insert_fft();

    std::cout << "@insert_fft after" << std::endl;

    // Move frames from gpu_space_transformation_buffer to
    // gpu_time_transformation_queue (with respect to
    // time_transformation_stride)
    insert_transfer_for_time_transformation();

    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    // !! BELOW ENQUEUE IN FN COMPUTE VECT MUST BE CONDITIONAL PUSH BACK !!
    // !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    // time transform
    if (cd_.time_transformation == TimeTransformation::STFT)
    {
        fourier_transforms_->insert_stft();
    }
    else if (cd_.time_transformation == TimeTransformation::PCA)
    {
        fourier_transforms_->insert_eigenvalue_filter();
    }

    fourier_transforms_->insert_time_transformation_cuts_view();

    // Used for phase increase
    fourier_transforms_->insert_store_p_frame();

    converts_->insert_to_float(unwrap_2d_requested_);

    postprocess_->insert_convolution();
    postprocess_->insert_renormalize();

    image_accumulation_->insert_image_accumulation();

    rendering_->insert_fft_shift();
    rendering_->insert_chart();
    rendering_->insert_log();

    insert_request_autocontrast();
    rendering_->insert_contrast(autocontrast_requested_,
                                autocontrast_slice_xz_requested_,
                                autocontrast_slice_yz_requested_);

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
    fn_compute_vect_.conditional_push_back([=]() { cudaXStreamSynchronize(stream_); });

    // Must be the last inserted function
    insert_reset_batch_index();
}

void Pipe::insert_wait_frames()
{
    fn_compute_vect_.push_back([&]() {
        // Wait while the input queue is enough filled
        while (gpu_input_queue_.get_size() < cd_.batch_size)
            continue;
    });
}

void Pipe::insert_reset_batch_index()
{
    fn_compute_vect_.conditional_push_back(
        [&]() { batch_env_.batch_index = 0; });
}

void Pipe::insert_transfer_for_time_transformation()
{
    fn_compute_vect_.push_back([&]() {
        time_transformation_env_.gpu_time_transformation_queue
            ->enqueue_multiple(buffers_.gpu_spatial_transformation_buffer.get(),
                               cd_.batch_size,
                               stream_);
        batch_env_.batch_index += cd_.batch_size;
        assert(batch_env_.batch_index <= cd_.time_transformation_stride);
    });
}

void Pipe::safe_enqueue_output(Queue& output_queue,
                               unsigned short* frame,
                               const std::string& error)
{
    if (!output_queue.enqueue(frame, stream_))
        throw CustomException(error, error_kind::fail_enqueue);
}

void Pipe::insert_output_enqueue_raw_mode()
{
    fn_compute_vect_.push_back([&]() {
        ++processed_output_fps_;

        safe_enqueue_output(
            gpu_output_queue_,
            static_cast<unsigned short*>(gpu_input_queue_.get_start()),
            "Can't enqueue the input frame in gpu_output_queue");

        gpu_input_queue_.dequeue();
    });
}

void Pipe::insert_output_enqueue_hologram_mode()
{
    fn_compute_vect_.conditional_push_back([&]() {
        ++processed_output_fps_;

        safe_enqueue_output(
            gpu_output_queue_,
            buffers_.gpu_output_frame.get(),
            "Can't enqueue the output frame in gpu_output_queue");

        // Always enqueue the cuts if enabled
        if (cd_.time_transformation_cuts_enabled)
        {
            safe_enqueue_output(
                *time_transformation_env_.gpu_output_queue_xz.get(),
                buffers_.gpu_output_frame_xz.get(),
                "Can't enqueue the output xz frame in output xz queue");

            safe_enqueue_output(
                *time_transformation_env_.gpu_output_queue_yz.get(),
                buffers_.gpu_output_frame_yz.get(),
                "Can't enqueue the output yz frame in output yz queue");
        }
    });
}

void Pipe::insert_raw_view()
{
    if (cd_.raw_view_enabled)
    {
        fn_compute_vect_.push_back([&]() {
            gpu_input_queue_.copy_multiple(*get_raw_view_queue(),
                                           cd_.batch_size,
                                           stream_);
        });
    }
}

void Pipe::insert_raw_record()
{
    if (cd_.frame_record_enabled && frame_record_env_.raw_record_enabled)
    {
        fn_compute_vect_.push_back([&]() {
            if (frame_record_env_.remaining_frames_to_record.has_value() &&
                frame_record_env_.remaining_frames_to_record.value() == 0)
                return;

            unsigned int nb_frames_to_transfer = 1;

            if (cd_.compute_mode == Computation::Hologram)
            {
                if (frame_record_env_.remaining_frames_to_record ==
                    std::nullopt)
                {
                    nb_frames_to_transfer =
                        static_cast<unsigned int>(cd_.batch_size.load());
                }
                else
                {
                    nb_frames_to_transfer = std::min(
                        static_cast<unsigned int>(cd_.batch_size.load()),
                        frame_record_env_.remaining_frames_to_record.value());
                }
            }

            gpu_input_queue_.copy_multiple(
                *frame_record_env_.gpu_frame_record_queue_,
                nb_frames_to_transfer,
                stream_);

            if (frame_record_env_.remaining_frames_to_record.has_value())
                frame_record_env_.remaining_frames_to_record.value() -=
                    nb_frames_to_transfer;
        });
    }
}

void Pipe::insert_hologram_record()
{
    if (cd_.frame_record_enabled && !frame_record_env_.raw_record_enabled)
    {
        fn_compute_vect_.conditional_push_back([&]() {
            if (frame_record_env_.remaining_frames_to_record.has_value() &&
                frame_record_env_.remaining_frames_to_record.value() == 0)
                return;

            if (gpu_output_queue_.get_fd().depth == 6)
                frame_record_env_.gpu_frame_record_queue_->enqueue_from_48bit(
                    buffers_.gpu_output_frame.get(),
                    stream_);
            else
                frame_record_env_.gpu_frame_record_queue_->enqueue(
                    buffers_.gpu_output_frame.get(),
                    stream_);

            if (frame_record_env_.remaining_frames_to_record.has_value())
                frame_record_env_.remaining_frames_to_record.value() -= 1;
        });
    }
}

void Pipe::insert_request_autocontrast()
{
    if (cd_.contrast_enabled && cd_.contrast_auto_refresh)
        request_autocontrast(cd_.current_window);
}

void Pipe::exec()
{
    if (global::global_config.flush_on_refresh)
        gpu_input_queue_.clear();

    Holovibes::instance().get_info_container().add_processed_fps(
        InformationContainer::FpsType::OUTPUT_FPS,
        processed_output_fps_);

    while (!termination_requested_)
    {
        try
        {
            if (gpu_input_queue_.get_size() >= 1)
            {
                // Run the entire pipeline of calculation
                run_all();

                if (refresh_requested_)
                    refresh();
            }
        }
        catch (CustomException& e)
        {
            pipe_error(1, e);
        }
    }

    Holovibes::instance().get_info_container().remove_processed_fps(
        InformationContainer::FpsType::OUTPUT_FPS);
}

std::unique_ptr<Queue>& Pipe::get_lens_queue()
{
    return fourier_transforms_->get_lens_queue();
}

void Pipe::insert_fn_end_vect(std::function<void()> function)
{
    std::lock_guard<std::mutex> lock(fn_end_vect_mutex_);
    fn_end_vect_.push_back(function);
}

void Pipe::autocontrast_end_pipe(WindowKind kind)
{
    insert_fn_end_vect([this, kind]() { request_autocontrast(kind); });
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