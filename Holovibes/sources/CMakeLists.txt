target_sources("${PROJECT_NAME}_lib" PRIVATE
    camera_dll.cc
    compute_settings.cc
    gpib_dll.cc
    options_parser.cc
    function_vector.cc
    queue.cc
    batch_input_queue.cc

    state/global_state_holder.cc

    API/contrast_API.cc
    API/CUDA_API.cc
    API/display_API.cc

    core/gsh_cache_on_change/common_on_change.cc
    core/gsh_cache_on_change/compute_gsh_on_change.cc
    core/gsh_cache_on_change/import_gsh_on_change.cc
    core/gsh_cache_on_change/view_gsh_on_change.cc

    core/pipe_request_on_sync_functions/compute_pipe_request_on_sync.cc
    core/pipe_request_on_sync_functions/export_pipe_request_on_sync.cc
    core/pipe_request_on_sync_functions/import_pipe_request_on_sync.cc
    core/pipe_request_on_sync_functions/view_pipe_request_on_sync.cc

    compute/compute_bundles.cc
    compute/compute_bundles_2d.cc
    compute/converts.cc
    compute/fourier_transform.cc
    compute/postprocessing.cc
    compute/rendering.cc
    compute/image_accumulation.cc
    compute/svd.cc

    core/holovibes.cc
    core/icompute.cc
    core/pipe.cc

    io_files/file_exception.cc
    io_files/input_frame_file.cc
    io_files/input_cine_file.cc
    io_files/input_holo_file.cc
    io_files/input_frame_file_factory.cc
    io_files/frame_file.cc
    io_files/output_avi_file.cc
    io_files/output_mp4_file.cc
    io_files/output_holo_file.cc
    io_files/output_frame_file_factory.cc
    io_files/output_frame_file.cc

    thread/batch_gpib_worker.cc
    thread/camera_frame_read_worker.cc
    thread/chart_record_worker.cc
    thread/compute_worker.cc
    thread/file_frame_read_worker.cc
    thread/frame_read_worker.cc
    thread/frame_record_worker.cc
    thread/information_worker.cc
    thread/worker.cc

    tools/cuda_tools/cublas_handle.cc
    tools/cuda_tools/cufft_handle.cc
    tools/cuda_tools/cusolver_handle.cc
    tools/hardware_limits.cc
    tools/power_of_two.cc
    tools/tools.cc
    tools/units/conversion_data.cc
    tools/units/fd_pixel.cc
    tools/units/opengl_position.cc
    tools/units/real_position.cc
    tools/units/window_pixel.cc
    tools/tools_debug.cc
    tools/logger/logger.cc
    tools/logger/thread_name_flag.cc

    state/global_state_holder.cc

    struct/all_struct.cc
)
