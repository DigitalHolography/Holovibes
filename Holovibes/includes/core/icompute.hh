/*! \file
 *
 * \brief Stores functions helping the editing of the images.
 */
#pragma once

#include <atomic>
#include <memory>

#include "config.hh"
#include "rect.hh"
#include "observable.hh"
#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "cufft_handle.hh"
#include "chart_point.hh"
#include "concurrent_deque.hh"
#include "enum_window_kind.hh"

namespace holovibes
{
class Queue;
class ComputeDescriptor;
class BatchInputQueue;
} // namespace holovibes

namespace holovibes
{
/*! \struct CoreBuffersEnv
 *
 * \brief Struct containing main buffers used by the pipe.
 */
struct CoreBuffersEnv
{
    /*! \brief Input buffer. Contains only one frame. We fill it with the input frame */
    cuda_tools::UniquePtr<cufftComplex> gpu_spatial_transformation_buffer = nullptr;

    /*! \brief Float buffer. Contains only one frame.
     *
     * We fill it with the correct computed p frame converted to float.
     */
    cuda_tools::UniquePtr<float> gpu_postprocess_frame = nullptr;
    /*! \brief Size in components (size in byte / sizeof(float)) of the gpu_postprocess_frame.
     *
     * Could be removed by changing gpu_postprocess_frame type to cuda_tools::Array.
     */
    unsigned int gpu_postprocess_frame_size = 0;
    /*! \brief Float XZ buffer of 1 frame, filled with the correct computer p XZ frame. */
    cuda_tools::UniquePtr<float> gpu_postprocess_frame_xz = nullptr;
    /*! \brief Float YZ buffer of 1 frame, filled with the correct computed p YZ frame. */
    cuda_tools::UniquePtr<float> gpu_postprocess_frame_yz = nullptr;

    /*! \brief Unsigned Short output buffer of 1 frame, inserted after all postprocessing on float_buffer */
    cuda_tools::UniquePtr<unsigned short> gpu_output_frame = nullptr;
    /*! \brief Unsigned Short XZ output buffer of 1 frame, inserted after all postprocessing on float_buffer_cut_xz */
    cuda_tools::UniquePtr<unsigned short> gpu_output_frame_xz = nullptr;
    /*! \brief Unsigned Short YZ output buffer of 1 frame, inserted after all postprocessing on float_buffer_cut_yz */
    cuda_tools::UniquePtr<unsigned short> gpu_output_frame_yz = nullptr;

    /*! \brief Contains only one frame used only for convolution */
    cuda_tools::UniquePtr<float> gpu_convolution_buffer = nullptr;

    /*! \brief Complex filter2d frame used to store the output_frame */
    cuda_tools::UniquePtr<cufftComplex> gpu_complex_filter2d_frame = nullptr;
    /*! \brief Float Filter2d frame used to store the gpu_complex_filter2d_frame */
    cuda_tools::UniquePtr<float> gpu_float_filter2d_frame = nullptr;
    /*! \brief Filter2d frame used to store the gpu_float_filter2d_frame */
    cuda_tools::UniquePtr<unsigned short> gpu_filter2d_frame = nullptr;
    /*! \brief Filter2d mask applied to gpu_spatial_transformation_buffer */
    cuda_tools::UniquePtr<float> gpu_filter2d_mask = nullptr;
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
     * The variable is incremented until it reachs batch_size in
     *enqueue_multiple, then it is set back to 0
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
     * the frame counter is equal to time_transformation_stride.
     */
    std::unique_ptr<Queue> gpu_time_transformation_queue = nullptr;
    /*! \brief STFT buffer.  Contains the result of the STFT done on the STFT queue.
     *
     * Contains time_transformation_size frames.
     */
    cuda_tools::UniquePtr<cufftComplex> gpu_p_acc_buffer = nullptr;
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
    cuda_tools::UniquePtr<cufftComplex> gpu_p_frame;

    /*! \name PCA time transformation
     * \{
     */
    cuda_tools::UniquePtr<cuComplex> pca_cov = nullptr;
    cuda_tools::UniquePtr<float> pca_eigen_values = nullptr;
    cuda_tools::UniquePtr<int> pca_dev_info = nullptr;
    /*! \} */
};

/*! \struct FrameRecordEnv
 *
 * \brief #TODO Add a description for this struct
 */
struct FrameRecordEnv
{
    std::unique_ptr<Queue> gpu_frame_record_queue_ = nullptr;
    bool raw_record_enabled = false;
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
    cuda_tools::UniquePtr<float> gpu_float_average_xy_frame = nullptr;
    /*! \brief Queue accumulating the XY computed frames. */
    std::unique_ptr<Queue> gpu_accumulation_xy_queue = nullptr;

    /*! \brief Frame to temporaly store the average on XZ view */
    cuda_tools::UniquePtr<float> gpu_float_average_xz_frame = nullptr;
    /*! \brief Queue accumulating the XZ computed frames. */
    std::unique_ptr<Queue> gpu_accumulation_xz_queue = nullptr;

    /*! \brief Frame to temporaly store the average on YZ axis */
    cuda_tools::UniquePtr<float> gpu_float_average_yz_frame = nullptr;
    /*! \brief Queue accumulating the YZ computed frames. */
    std::unique_ptr<Queue> gpu_accumulation_yz_queue = nullptr;
};

/*! \class ICompute
 *
 * \brief Stores functions helping the editing of the images.
 *
 * Stores all the functions that will be used before doing any sort
 * of editing to the image (i.e. refresh functions or caller).
 */
// #TODO Add \name tags between groups of methods and attributes to make the documentation clearer
class ICompute : public Observable
{
    friend class ThreadCompute;

  public:
    ICompute(BatchInputQueue& input, Queue& output, ComputeDescriptor& cd, const cudaStream_t& stream);
    void request_refresh();
    void request_output_resize(unsigned int new_output_size);
    void request_autocontrast(WindowKind kind);
    void request_update_time_transformation_size();
    void request_update_unwrap_size(const unsigned size);
    void request_unwrapping_1d(const bool value);
    void request_unwrapping_2d(const bool value);
    void request_display_chart();
    void request_disable_display_chart();
    void request_record_chart(unsigned int nb_chart_points_to_record);
    void request_disable_record_chart();
    void request_termination();
    void request_update_batch_size();
    void request_update_time_transformation_stride();
    void request_disable_lens_view();
    void request_raw_view();
    void request_disable_raw_view();
    void request_filter2d_view();
    void request_disable_filter2d_view();
    void request_hologram_record(std::optional<unsigned int> nb_frames_to_record);
    void request_raw_record(std::optional<unsigned int> nb_frames_to_record);
    void request_disable_frame_record();
    void request_clear_img_acc();
    void request_convolution();
    void request_disable_convolution();

    /*! \brief Execute one iteration of the ICompute.
     *
     * Checks the number of frames in input queue that must at least time_transformation_size.
     * Call each function of the ICompute.
     * Enqueue the output frame contained in gpu_output_buffer.
     * Dequeue one frame of the input queue.
     * Check if a ICompute refresh has been requested.
     *
     * The ICompute can not be interrupted for parameters changes until the refresh method is called.
     */
    virtual void exec() = 0;

    void create_stft_slice_queue();
    void delete_stft_slice_queue();
    std::unique_ptr<Queue>& get_stft_slice_queue(int i);
    bool get_cuts_request();
    bool get_cuts_delete_request();
    bool get_request_refresh();

    bool get_unwrap_1d_request() const { return unwrap_1d_requested_; }
    bool get_unwrap_2d_request() const { return unwrap_2d_requested_; }
    bool get_autocontrast_request() const { return autocontrast_requested_; }
    bool get_autocontrast_slice_xz_request() const { return autocontrast_slice_xz_requested_; }
    bool get_autocontrast_slice_yz_request() const { return autocontrast_slice_yz_requested_; }
    bool get_refresh_request() const { return refresh_requested_; }
    bool get_update_time_transformation_size_request() const { return update_time_transformation_size_requested_; }
    bool get_stft_update_roi_request() const { return stft_update_roi_requested_; }
    bool get_termination_request() const { return termination_requested_; }
    bool get_request_time_transformation_cuts() const { return request_time_transformation_cuts_; }
    bool get_request_delete_time_transformation_cuts() const { return request_delete_time_transformation_cuts_; }
    std::optional<unsigned int> get_output_resize_request() const { return output_resize_requested_; }
    bool get_raw_view_requested() const { return raw_view_requested_; }
    bool get_disable_raw_view_requested() const { return disable_raw_view_requested_; }
    bool get_filter2d_view_requested() const { return filter2d_view_requested_; }
    bool get_disable_filter2d_view_requested() const { return disable_filter2d_view_requested_; }
    bool get_chart_display_requested() const { return chart_display_requested_; }
    std::optional<unsigned int> get_chart_record_requested() const { return chart_record_requested_; }
    bool get_disable_chart_display_requested() const { return disable_chart_display_requested_; }
    bool get_disable_chart_record_requested() const { return disable_chart_record_requested_; }
    std::optional<std::optional<unsigned int>> get_hologram_record_requested() const
    {
        return hologram_record_requested_;
    }
    std::optional<std::optional<unsigned int>> get_raw_record_requested() const { return raw_record_requested_; }
    bool get_disable_frame_record_requested() const { return disable_frame_record_requested_; }
    bool get_convolution_requested() const { return convolution_requested_; }
    bool get_disable_convolution_requested() const { return convolution_requested_; }

    virtual std::unique_ptr<Queue>& get_lens_queue() = 0;

    virtual std::unique_ptr<Queue>& get_raw_view_queue();

    virtual std::unique_ptr<Queue>& get_filter2d_view_queue();

    virtual std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_display_queue();

    virtual std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_record_queue();

    virtual std::unique_ptr<Queue>& get_frame_record_queue();

  protected:
    virtual void refresh() = 0;
    virtual void pipe_error(const int& err_count, std::exception& e);
    virtual bool update_time_transformation_size(const unsigned short time_transformation_size);

    /*! \name Resources management
     * \{
     */
    virtual void update_spatial_transformation_parameters();
    void init_cuts();
    void dispose_cuts();
    /*! \} */

    ICompute& operator=(const ICompute&) = delete;
    ICompute(const ICompute&) = delete;

  protected:
    /*! \brief Compute Descriptor. */
    ComputeDescriptor& cd_;

    /*! \brief Reference on the input queue, owned by MainWindow. */
    BatchInputQueue& gpu_input_queue_;
    /*! \brief Reference on the output queue, owned by MainWindow. */
    Queue& gpu_output_queue_;

    /*! \brief Main buffers. */
    CoreBuffersEnv buffers_;

    /*! \brief Batch environment */
    BatchEnv batch_env_;

    /*! \brief STFT environment. */
    TimeTransformationEnv time_transformation_env_;

    /*! \brief Frame Record environment (Raw + Hologram) */
    FrameRecordEnv frame_record_env_;

    /*! \brief Chart environment. */
    ChartEnv chart_env_;

    /*! \brief Image accumulation environment */
    ImageAccEnv image_acc_env_;

    /*! \brief Queue storing raw frames used by raw view */
    std::unique_ptr<Queue> gpu_raw_view_queue_{nullptr};

    /*! \brief Queue storing filter2d frames */
    std::unique_ptr<Queue> gpu_filter2d_view_queue_{nullptr};

    /*! \brief Pland 2D. Used for spatial fft performed on the complex input frame. */
    cuda_tools::CufftHandle spatial_transformation_plan_;

    /*! \brief Pland 2D. Used for unwrap 2D. */
    cuda_tools::CufftHandle plan_unwrap_2d_;

    /*! \brief Compute stream to perform pipe computation */
    const cudaStream_t& stream_;

    /*! \brief Chrono counting time between two iteration
     *
     * Taking into account steps, since it is executing at the end of pipe.
     */
    std::chrono::time_point<std::chrono::steady_clock> past_time_;

    /*! \brief Counting pipe iteration, in order to update fps only every 100 iterations. */
    unsigned int frame_count_{0};

    std::atomic<bool> unwrap_1d_requested_{false};
    std::atomic<bool> unwrap_2d_requested_{false};
    std::atomic<bool> autocontrast_requested_{false};
    std::atomic<bool> autocontrast_slice_xz_requested_{false};
    std::atomic<bool> autocontrast_slice_yz_requested_{false};
    std::atomic<bool> autocontrast_filter2d_requested_{false};
    std::atomic<bool> refresh_requested_{false};
    std::atomic<bool> update_time_transformation_size_requested_{false};
    std::atomic<bool> stft_update_roi_requested_{false};
    std::atomic<bool> chart_display_requested_{false};
    std::atomic<bool> disable_chart_display_requested_{false};
    std::atomic<std::optional<unsigned int>> chart_record_requested_{std::nullopt};
    std::atomic<bool> disable_chart_record_requested_{false};
    std::atomic<std::optional<unsigned int>> output_resize_requested_{std::nullopt};
    std::atomic<bool> raw_view_requested_{false};
    std::atomic<bool> disable_raw_view_requested_{false};
    std::atomic<bool> filter2d_view_requested_{false};
    std::atomic<bool> disable_filter2d_view_requested_{false};
    std::atomic<bool> termination_requested_{false};
    std::atomic<bool> request_time_transformation_cuts_{false};
    std::atomic<bool> request_delete_time_transformation_cuts_{false};
    std::atomic<bool> request_update_batch_size_{false};
    std::atomic<bool> request_update_time_transformation_stride_{false};
    std::atomic<bool> request_disable_lens_view_{false};
    std::atomic<std::optional<std::optional<unsigned int>>> hologram_record_requested_{std::nullopt};
    std::atomic<std::optional<std::optional<unsigned int>>> raw_record_requested_{std::nullopt};
    std::atomic<bool> disable_frame_record_requested_{false};
    std::atomic<bool> request_clear_img_acc_{false};
    std::atomic<bool> convolution_requested_{false};
    std::atomic<bool> disable_convolution_requested_{false};
};
} // namespace holovibes
