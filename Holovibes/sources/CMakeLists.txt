target_sources("${PROJECT_NAME}_lib" PRIVATE
    camera_dll.cc
    config.cc
    gpib_dll.cc
    options_parser.cc
    function_vector.cc
    queue.cc
    recorder.cc

    compute/compute_bundles.cc
    compute/compute_bundles_2d.cc
    compute/compute_descriptor.cc
    compute/converts.cc
    compute/fourier_transform.cc
    compute/postprocessing.cc
    compute/rendering.cc
    compute/image_accumulation.cc

    core/holovibes.cc
    core/icompute.cc
    core/pipe.cc

    gui/other/gui_curve_plot.cc
    gui/other/gui_drag_drop_lineedit.cc
    gui/other/gui_group_box.cc
    gui/selections/interfaces/Overlay.cc
    gui/selections/interfaces/rect_overlay.cc
    gui/selections/interfaces/square_overlay.cc
    gui/selections/overlays/composite_area_overlay.cc
    gui/selections/overlays/cross_overlay.cc
    gui/selections/overlays/filter2d_overlay.cc
    gui/selections/overlays/filter2d_subzone_overlay.cc
    gui/selections/overlays/noise_overlay.cc
    gui/selections/overlays/rainbow_overlay.cc
    gui/selections/overlays/reticle_overlay.cc
    gui/selections/overlays/signal_overlay.cc
    gui/selections/overlays/slice_cross_overlay.cc
    gui/selections/overlays/zoom_overlay.cc
    gui/selections/overlay_manager.cc
    gui/windows/BasicOpenGLWindow.cc
    gui/windows/RawWindow.cc
    gui/windows/HoloWindow.cc
    gui/windows/MainWindow.cc
    gui/windows/PlotWindow.cc
    gui/windows/SliceWindow.cc

    io_files/input_file.cc
    io_files/input_cine_file.cc
    io_files/input_holo_file.cc
    io_files/input_file_handler.cc
    io_files/io_file.cc
    io_files/output_holo_file.cc
    io_files/output_file_handler.cc

    thread/ithread_input.cc
    thread/thread_capture.cc
    thread/thread_compute.cc
    thread/thread_csv_record.cc
    thread/thread_reader.cc
    thread/thread_recorder.cc
    thread/thread_timer.cc

    tools/cuda_tools/cublas_handle.cc
    tools/cuda_tools/cufft_handle.cc
    tools/cuda_tools/cusolver_handle.cc
    tools/cuda_tools/nppi_data.cc
    tools/hardware_limits.cc
    tools/info_manager.cc
    tools/power_of_two.cc
    tools/tools.cc
    tools/units/conversion_data.cc
    tools/units/fd_pixel.cc
    tools/units/opengl_position.cc
    tools/units/real_position.cc
    tools/units/window_pixel.cc
    tools/tools_debug.cc
)
