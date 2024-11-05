#pragma once

#include "queue.hh"
#include "unique_ptr.hh"
#include "cufft_handle.hh"
#include "chart_point.hh"
#include "concurrent_deque.hh"

namespace holovibes
{
/*! \struct CoreBuffersEnv
 *
 * \brief Struct containing main buffers used by the pipe.
 */
struct CoreBuffersEnv
{
    /*! \brief Input buffer. Contains only one frame. We fill it with the input frame */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_spatial_transformation_buffer = nullptr;

    /*! \brief Float buffer. Contains only one frame.
     *
     * We fill it with the correct computed p frame converted to float.
     */
    cuda_tools::CudaUniquePtr<float> gpu_postprocess_frame = nullptr;

    /*! \brief Size in components (size in byte / sizeof(float)) of the gpu_postprocess_frame.
     *
     * Could be removed by changing gpu_postprocess_frame type to cuda_tools::Array.
     */
    unsigned int gpu_postprocess_frame_size = 0;

    /*! \brief Float XZ buffer of 1 frame, filled with the correct computed p XZ frame. */
    cuda_tools::CudaUniquePtr<float> gpu_postprocess_frame_xz = nullptr;

    /*! \brief Float YZ buffer of 1 frame, filled with the correct computed p YZ frame. */
    cuda_tools::CudaUniquePtr<float> gpu_postprocess_frame_yz = nullptr;

    /*! \brief Unsigned Short output buffer of 1 frame, inserted after all postprocessing on float_buffer */
    cuda_tools::CudaUniquePtr<unsigned short> gpu_output_frame = nullptr;

    /*! \brief Unsigned Short XZ output buffer of 1 frame, inserted after all postprocessing on float_buffer_cut_xz */
    cuda_tools::CudaUniquePtr<unsigned short> gpu_output_frame_xz = nullptr;

    /*! \brief Unsigned Short YZ output buffer of 1 frame, inserted after all postprocessing on float_buffer_cut_yz */
    cuda_tools::CudaUniquePtr<unsigned short> gpu_output_frame_yz = nullptr;

    /*! \brief Contains only one frame used only for convolution */
    cuda_tools::CudaUniquePtr<float> gpu_convolution_buffer = nullptr;

    /*! \brief Complex filter2d frame used to store the output_frame */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_complex_filter2d_frame = nullptr;

    /*! \brief Float Filter2d frame used to store the gpu_complex_filter2d_frame */
    cuda_tools::CudaUniquePtr<float> gpu_float_filter2d_frame = nullptr;

    /*! \brief Filter2d frame used to store the gpu_float_filter2d_frame */
    cuda_tools::CudaUniquePtr<unsigned short> gpu_filter2d_frame = nullptr;

    /*! \brief Filter2d mask applied to gpu_spatial_transformation_buffer */
    cuda_tools::CudaUniquePtr<float> gpu_filter2d_mask = nullptr;

    /*! \brief InputFilter mask */
    cuda_tools::CudaUniquePtr<float> gpu_input_filter_mask = nullptr;
};

/*! \struct MomentsEnv
 *
 *  \brief Struct containing buffers used for the computation of the moments
 *
 */
struct MomentsEnv
{
    /*! \brief Contains the moments, computed from the frequencies resulting from the stft and the initial batch of
     * frames.
     *
     * The moment of order 0 is equal to the batch of frames multiplied by the vector f of frequencies at
     * Contains time_transformation_size frames.
     */
    cuda_tools::CudaUniquePtr<float> moment0_buffer = nullptr;
    cuda_tools::CudaUniquePtr<float> moment1_buffer = nullptr;
    cuda_tools::CudaUniquePtr<float> moment2_buffer = nullptr;

    /*! \brief Temporary buffer that contains a batch of time transformation size frames
     *  It will contains the complex modulus of result of the time transformation
     */
    cuda_tools::CudaUniquePtr<float> stft_res_buffer = nullptr;

    /*! \brief Vector of size time_transformation_size filled with 1, representing the frequencies at order 0.
     * Used to compute the moment of order 0*/
    cuda_tools::CudaUniquePtr<float> f0_buffer = nullptr;
    /*! \brief Vector of size time_transformation_size, representing the frequencies at order 1.
     * Used to compute the moment of order 1*/
    cuda_tools::CudaUniquePtr<float> f1_buffer = nullptr;
    /*! \brief Vector of size time_transformation_size, representing the frequencies at order 2.
     * Used to compute the moment of order 2*/
    cuda_tools::CudaUniquePtr<float> f2_buffer = nullptr;

    /*! \brief Starts and end frequencies of calculus */
    unsigned short f_start;
    unsigned short f_end;
};

/*! \struct BatchEnv
 *
 * \brief Struct containing variables related to the batch in the pipe
 */
struct BatchEnv
{
    /*! \brief Current frames processed in the batch
     *
     * At index 0, batch_size frames are enqueued, spatial transformation is
     * also executed in batch
     * Batch size frames are enqueued in the gpu_time_transformation_queue
     * This is done for perfomances reasons
     *
     * The variable is incremented by batch_size until it reaches timestride in
     * enqueue_multiple, then it is set back to 0
     */
    uint batch_index = 0;
};

/*! \struct TimeTransformationEnv
 *
 * \brief Struct containing variables related to STFT shared by multiple
 * features of the pipe.
 */
struct TimeTransformationEnv
{
    /*! \brief STFT Queue. It accumulates input frames after spatial FFT.
     *
     * Contains time_transformation_size frames.
     * Frames are accumulated in order to apply STFT only when
     * the frame counter is equal to time_stride.
     */
    std::unique_ptr<Queue> gpu_time_transformation_queue = nullptr;

    /*! \brief STFT buffer. Contains the result of the STFT done on the STFT queue.
     *
     * Contains time_transformation_size frames.
     */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_p_acc_buffer = nullptr;

    /*! \brief STFT XZ Queue. Contains the ouput of the STFT on slice XZ.
     *
     * Enqueued with gpu_float_buffer or gpu_ushort_buffer.
     */
    std::unique_ptr<Queue> gpu_output_queue_xz = nullptr;

    /*! \brief STFT YZ Queue. Contains the ouput of the STFT on slice YZ.
     *
     * Enqueued with gpu_float_buffer or gpu_ushort_buffer.
     */
    std::unique_ptr<Queue> gpu_output_queue_yz = nullptr;

    /*! \brief Plan 1D used for the STFT. */
    cuda_tools::CufftHandle stft_plan;

    /*! \brief Hold the P frame after the time transformation computation. */
    cuda_tools::CudaUniquePtr<cufftComplex> gpu_p_frame;

    /*! \name PCA time transformation
     * \{
     */
    cuda_tools::CudaUniquePtr<cuComplex> pca_cov = nullptr;
    cuda_tools::CudaUniquePtr<float> pca_eigen_values = nullptr;
    cuda_tools::CudaUniquePtr<int> pca_dev_info = nullptr;
    /*! \} */
};

/*! \struct ChartEnv
 *
 * \brief Structure containing variables related to the chart display and
 * recording.
 */
struct ChartEnv
{
    std::unique_ptr<ConcurrentDeque<ChartPoint>> chart_display_queue_ = nullptr;
    std::unique_ptr<ConcurrentDeque<ChartPoint>> chart_record_queue_ = nullptr;
    unsigned int nb_chart_points_to_record_ = 0;
};

/*! \struct ImageAccEnv
 *
 * \brief #TODO Add a description for this struct
 */
struct ImageAccEnv
{
    /*! \brief Frame to temporaly store the average on XY view */
    cuda_tools::CudaUniquePtr<float> gpu_float_average_xy_frame = nullptr;

    /*! \brief Queue accumulating the XY computed frames. */
    std::unique_ptr<Queue> gpu_accumulation_xy_queue = nullptr;

    /*! \brief Frame to temporaly store the average on XZ view */
    cuda_tools::CudaUniquePtr<float> gpu_float_average_xz_frame = nullptr;

    /*! \brief Queue accumulating the XZ computed frames. */
    std::unique_ptr<Queue> gpu_accumulation_xz_queue = nullptr;

    /*! \brief Frame to temporaly store the average on YZ axis */
    cuda_tools::CudaUniquePtr<float> gpu_float_average_yz_frame = nullptr;

    /*! \brief Queue accumulating the YZ computed frames. */
    std::unique_ptr<Queue> gpu_accumulation_yz_queue = nullptr;
};

/*! \struct VesselnessMaskEnv
 *
 *  \brief Struct containing buffers used for the computation of the vesselness masks
 *
 */
struct VesselnessMaskEnv
{
    /*! \brief Get the mean image */
    cuda_tools::CudaUniquePtr<float> m0_ff_sum_image_ = nullptr;

    /*! \brief Buffer of size 'time_window_' TODO refaire ca to compute the mean of m0 imgs */
    cuda_tools::CudaUniquePtr<float> buffer_m0_ff_img_ = nullptr;

    /*! \brief image with mean calculated for a time window*/
    cuda_tools::CudaUniquePtr<float> image_with_mean_ = nullptr;

    /*! \brief image with mean and centered calculated for a time window*/
    cuda_tools::CudaUniquePtr<float> image_centered_ = nullptr;

    /*! \brief Gaussian kernels converted in cuComplex used in vesselness filter */
    cuda_tools::CudaUniquePtr<cuComplex> g_xx_mul_comp_ = nullptr;

    cuda_tools::CudaUniquePtr<cuComplex> g_xy_mul_comp_ = nullptr;

    cuda_tools::CudaUniquePtr<cuComplex> g_yy_mul_comp_ = nullptr;

    cuda_tools::CudaUniquePtr<cuComplex> temp_buffer_ = nullptr;

    cuda_tools::CudaUniquePtr<float> g_xx_mul_ = nullptr;

    cuda_tools::CudaUniquePtr<float> g_xy_mul_ = nullptr;

    cuda_tools::CudaUniquePtr<float> g_yy_mul_ = nullptr;

    /*! \brief Time window for mask */
    int time_window_;

    /*! \brief Get the number of image for the mean mask*/
    int number_image_mean_ = 0;

    /*! \brief X size of kernel */
    int kernel_x_size_ = 0;

    /*! \brief Y size of kernel */
    int kernel_y_size_ = 0;
};
} // namespace holovibes
