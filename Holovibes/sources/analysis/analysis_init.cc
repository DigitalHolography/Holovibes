#include "convolution.cuh"
#include "cuda_memory.cuh"
#include "shift_corners.cuh"
#include "holovibes.hh"
#include "matrix_operations.hh"
#include "moments_treatments.cuh"
#include "vesselness_filter.cuh"
#include "barycentre.cuh"
#include "vascular_pulse.cuh"
#include "tools_analysis.cuh"
#include "API.hh"
#include "otsu.cuh"
#include "cublas_handle.hh"
#include "bw_area.cuh"
#include "circular_video_buffer.hh"
#include "segment_vessels.cuh"
#include "tools_analysis_debug.hh"

#define DIAPHRAGM_FACTOR 0.4f
#define OTSU_BINS 256

namespace holovibes::analysis
{
void Analysis::init_params_vesselness_filter(float* result_transpose,
                                             float* target,
                                             float sigma,
                                             int x_size,
                                             int y_size,
                                             int x_lim,
                                             int y_lim,
                                             int p,
                                             int q,
                                             cudaStream_t stream)
{
    // Step 1: Initialize normalized centered lists (e.g., for x_size = 3: [-1, 0, 1])
    // Allocate memory for the x-axis normalized list
    float* x;
    cudaXMalloc(&x, x_size * sizeof(float));
    normalized_list(x, x_lim, x_size, stream);

    // Allocate memory for the y-axis normalized list
    float* y;
    cudaXMalloc(&y, y_size * sizeof(float));
    normalized_list(y, y_lim, y_size, stream);

    // Step 2: Compute X and Y derivative Gaussian kernels
    // Allocate memory and compute the derivative Gaussian kernel for the x-axis
    float* g_px;
    cudaXMalloc(&g_px, x_size * sizeof(float));
    comp_dgaussian(g_px, x, x_size, sigma, p, stream);

    // Allocate memory and compute the derivative Gaussian kernel for the y-axis
    float* g_qy;
    cudaXMalloc(&g_qy, y_size * sizeof(float));
    comp_dgaussian(g_qy, y, y_size, sigma, q, stream);

    // Step 3: Perform matrix multiplication using the computed kernels
    // Multiply g_qy and g_px to generate a matrix and store the result in `target`
    holovibes::compute::matrix_multiply<float>(g_qy, g_px, y_size, x_size, 1, target, cublas_handler_);

    // Step 4: Transpose the resulting matrix
    // Define scaling factors for the cublasSgeam operation
    const float alpha = 1.0f;
    const float beta = 0.0f;

    // Use cuBLAS to perform the matrix transpose (target -> result_transpose)
    cublasSafeCall(cublasSgeam(cublas_handler_,
                               CUBLAS_OP_T,      // Transpose operation
                               CUBLAS_OP_N,      // No transpose for the second matrix
                               x_size,           // Number of rows in the result
                               y_size,           // Number of columns in the result
                               &alpha,           // Scaling factor for `target`
                               target,           // Input matrix (to transpose)
                               y_size,           // Leading dimension of `target`
                               &beta,            // Scaling factor for a second (non-existent) matrix
                               nullptr,          // Second matrix (not used)
                               y_size,           // Leading dimension of the second matrix
                               result_transpose, // Output transposed matrix
                               x_size));         // Leading dimension of the output matrix

    // Step 5: Free allocated GPU memory
    // Deallocate the GPU memory for temporary arrays and kernels
    cudaXStreamSynchronize(stream_);
    cudaXFree(x);
    cudaXFree(y);
    cudaXFree(g_qy);
    cudaXFree(g_px);
}

void Analysis::init()
{
    // Step 1: Get frame resolution and initialize parameters for Gaussian kernels
    const size_t frame_res = fd_.get_frame_res();

    // Gaussian derivative kernel parameters
    float gdk_sigma = setting<settings::VesselnessSigma>(); // Sigma for Gaussian kernels
    int gamma = 1;
    int A = std::pow(gdk_sigma, gamma);   // Placeholder computation
    int x_lim = std::ceil(4 * gdk_sigma); // Limits for the kernel in x
    int y_lim = std::ceil(4 * gdk_sigma); // Limits for the kernel in y
    int x_size = x_lim * 2 + 1;           // Total size of the kernel in x
    int y_size = y_lim * 2 + 1;           // Total size of the kernel in y

    // Gaussian kernel parameters for vascular pulse processing
    float vp_sigma = 0.02 * fd_.width; // Sigma scaled to frame width

    // Step 2: Initialize buffers for binary image processing
    bw_area_env_.uint_buffer_1_.safe_resize(frame_res);
    bw_area_env_.uint_buffer_2_.safe_resize(frame_res);
    bw_area_env_.size_t_gpu_.resize(1);
    bw_area_env_.float_buffer_.safe_resize(frame_res);

    // Step 3: Allocate histogram buffer for Otsu thresholding
    otsu_env_.otsu_histo_buffer_.resize(OTSU_BINS);

    // Step 4: Allocate buffers for vesselness mask processing
    vesselness_mask_env_.time_window_ = api::get_time_window(); // Temporal window for processing
    vesselness_mask_env_.m0_ff_video_centered_.safe_resize(buffers_.gpu_postprocess_frame_size *
                                                           vesselness_mask_env_.time_window_);
    vesselness_mask_env_.vascular_image_.safe_resize(frame_res);
    vesselness_mask_env_.m1_divided_by_m0_frame_.safe_resize(frame_res);
    vesselness_mask_env_.circle_mask_.safe_resize(frame_res);
    vesselness_mask_env_.bwareafilt_result_.safe_resize(frame_res);
    vesselness_mask_env_.mask_vesselness_.safe_resize(frame_res);
    vesselness_mask_env_.mask_vesselness_clean_.safe_resize(frame_res);
    vesselness_mask_env_.quantizedVesselCorrelation_.safe_resize(frame_res);

    // Set kernel dimensions and compute vascular kernel size
    vesselness_mask_env_.kernel_x_size_ = x_size;
    vesselness_mask_env_.kernel_y_size_ = y_size;
    vesselness_mask_env_.vascular_kernel_size_ = 2 * std::ceil(2 * vp_sigma) + 1;

    // Allocate buffers for Gaussian derivative kernel computations
    vesselness_mask_env_.g_xx_mul_.safe_resize(x_size * y_size);
    vesselness_mask_env_.g_xy_mul_.safe_resize(x_size * y_size);
    vesselness_mask_env_.g_yy_mul_.safe_resize(x_size * y_size);
    vesselness_mask_env_.vascular_kernel_.safe_resize(vesselness_mask_env_.vascular_kernel_size_ *
                                                      vesselness_mask_env_.vascular_kernel_size_);

    vesselness_mask_env_.before_threshold.safe_resize(frame_res);
    vesselness_mask_env_.R_vascular_pulse_.safe_resize(frame_res);

    // Step 5: Initialize circular video buffers
    vesselness_mask_env_.m0_ff_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);
    vesselness_mask_env_.f_avg_video_cb_ =
        std::make_unique<CircularVideoBuffer>(frame_res, api::get_time_window(), stream_);

    // Step 6: Allocate buffers for vesselness filter processing
    vesselness_filter_struct_.I.safe_resize(frame_res);
    vesselness_filter_struct_.convolution_tmp_buffer.safe_resize(frame_res);
    vesselness_filter_struct_.H.safe_resize(frame_res * 3); // For Hessian matrix elements
    vesselness_filter_struct_.lambda_1.safe_resize(frame_res);
    vesselness_filter_struct_.lambda_2.safe_resize(frame_res);
    vesselness_filter_struct_.R_blob.safe_resize(frame_res);
    vesselness_filter_struct_.c_temp.safe_resize(frame_res);
    vesselness_filter_struct_.CRV_circle_mask.safe_resize(frame_res);
    vesselness_filter_struct_.vascular_pulse.safe_resize(vesselness_mask_env_.time_window_);
    vesselness_filter_struct_.vascular_pulse_centered.safe_resize(vesselness_mask_env_.time_window_);
    vesselness_filter_struct_.std_M0_ff_video_centered.safe_resize(buffers_.gpu_postprocess_frame_size);
    vesselness_filter_struct_.std_vascular_pulse_centered.safe_resize(1);
    vesselness_filter_struct_.thresholds.safe_resize(4);

    // Step 7: Allocate buffers for the first mask choroid
    first_mask_choroid_struct_.first_mask_choroid.safe_resize(frame_res);

    // Step 8: Compute Gaussian derivative kernels and transpose them
    float* result_transpose;

    // Compute g_xx kernel
    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    init_params_vesselness_filter(result_transpose,
                                  vesselness_mask_env_.g_xx_mul_,
                                  gdk_sigma,
                                  x_size,
                                  y_size,
                                  x_lim,
                                  y_lim,
                                  2,
                                  0,
                                  stream_);
    vesselness_mask_env_.g_xx_mul_.reset(result_transpose);

    // Compute g_xy kernel (note: g_yx is the same as g_xy)
    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    init_params_vesselness_filter(result_transpose,
                                  vesselness_mask_env_.g_xy_mul_,
                                  gdk_sigma,
                                  x_size,
                                  y_size,
                                  x_lim,
                                  y_lim,
                                  1,
                                  1,
                                  stream_);
    vesselness_mask_env_.g_xy_mul_.reset(result_transpose);

    // Compute g_yy kernel
    cudaXMalloc(&result_transpose, sizeof(float) * x_size * y_size);
    init_params_vesselness_filter(result_transpose,
                                  vesselness_mask_env_.g_yy_mul_,
                                  gdk_sigma,
                                  x_size,
                                  y_size,
                                  x_lim,
                                  y_lim,
                                  0,
                                  2,
                                  stream_);
    vesselness_mask_env_.g_yy_mul_.reset(result_transpose);

    // Step 9: Compute Gaussian kernel for vascular pulse
    compute_gauss_kernel(vesselness_mask_env_.vascular_kernel_, vp_sigma, stream_);

    // Step 10: Allocate the final result buffer and set it to 0
    mask_result_buffer_.safe_resize(frame_res);
    cudaXMemset(mask_result_buffer_, 0, frame_res * sizeof(float));

    // Other
    chart_mean_vessels_env_.float_buffer_gpu_.resize(8);
    if (chart_mean_vessels_env_.chart_display_queue_.get() == nullptr)
        chart_mean_vessels_env_.chart_display_queue_.reset(new ConcurrentDeque<ChartMeanVesselsPoint>());
}
} // namespace holovibes::analysis