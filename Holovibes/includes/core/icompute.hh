/*! \file
 *
 * \brief Stores functions helping the editing of the images.
 */
#pragma once

#include "env_structs.hh"

namespace holovibes
{

/*! \class ICompute
 *
 * \brief Stores functions helping the editing of the images.
 *
 * Stores all the functions that will be used before doing any sort
 * of editing to the image (i.e. refresh functions or caller).
 */
// #TODO Add \name tags between groups of methods and attributes to make the documentation clearer
class ICompute
{
  public:
    ICompute(BatchInputQueue& input, Queue& output, const cudaStream_t& stream);
    ICompute& operator=(const ICompute&) = delete;
    ICompute(const ICompute&) = delete;
    virtual ~ICompute();

  public:
    BatchInputQueue& get_gpu_input_queue() { return gpu_input_queue_; };
    Queue& gpu_output_queue() { return gpu_output_queue_; }
    CoreBuffersEnv& get_buffers() { return buffers_; }
    BatchEnv& get_batch_env() { return batch_env_; }
    TimeTransformationEnv& get_time_transformation_env() { return time_transformation_env_; }
    FrameRecordEnv& get_frame_record_env() { return frame_record_env_; }
    ChartEnv& get_chart_env() { return chart_env_; }
    ImageAccEnv& get_image_acc_env() { return image_acc_env_; }

    AdvancedCache::Cache& get_advanced_cache() { return advanced_cache_; }
    ComputeCache::Cache& get_compute_cache() { return compute_cache_; }
    ExportCache::Cache& get_export_cache() { return export_cache_; }
    CompositeCache::Cache& get_composite_cache() { return composite_cache_; }
    Filter2DCache::Cache& get_filter2d_cache() { return filter2d_cache_; }
    ViewCache::Cache& get_view_cache() { return view_cache_; }
    ZoneCache::Cache& get_zone_cache() { return zone_cache_; }
    RequestCache::Cache& get_unknown_cache() { return unknown_cache_; }

    std::unique_ptr<Queue>& get_raw_view_queue();
    std::unique_ptr<Queue>& get_filter2d_view_queue();
    std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_display_queue();
    std::unique_ptr<ConcurrentDeque<ChartPoint>>& get_chart_record_queue();
    std::unique_ptr<Queue>& get_frame_record_queue();

  public:
    void request_refresh();

    void request_output_resize(unsigned int new_output_size);
    void request_autocontrast(WindowKind kind);
    void request_unwrapping_1d(const bool value);
    void request_unwrapping_2d(const bool value);
    void request_display_chart();
    void request_disable_display_chart();
    void request_record_chart(unsigned int nb_chart_points_to_record);
    void request_disable_record_chart();
    void request_termination();
    void request_update_time_stride();
    void request_disable_lens_view();
    void request_filter2d_view();
    void request_disable_filter2d_view();
    void request_hologram_record();
    void request_raw_record();
    void request_cuts_record(RecordMode rm);

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

    bool get_unwrap_1d_request() const { return unwrap_1d_requested_; }
    bool get_unwrap_2d_request() const { return unwrap_2d_requested_; }
    bool get_autocontrast_request() const { return autocontrast_requested_; }
    bool get_autocontrast_slice_xz_request() const { return autocontrast_slice_xz_requested_; }
    bool get_autocontrast_slice_yz_request() const { return autocontrast_slice_yz_requested_; }
    bool get_refresh_request() const { return refresh_requested_; }
    bool get_stft_update_roi_request() const { return stft_update_roi_requested_; }
    bool get_termination_request() const { return termination_requested_; }

    std::optional<unsigned int> get_output_resize_request() const { return output_resize_requested_; }
    bool get_disable_lens_view_requested() const { return request_disable_lens_view_; }
    bool get_filter2d_view_requested() const { return filter2d_view_requested_; }
    bool get_disable_filter2d_view_requested() const { return disable_filter2d_view_requested_; }
    bool get_chart_display_requested() const { return chart_display_requested_; }
    std::optional<unsigned int> get_chart_record_requested() const { return chart_record_requested_; }
    bool get_disable_chart_display_requested() const { return disable_chart_display_requested_; }
    bool get_disable_chart_record_requested() const { return disable_chart_record_requested_; }
    bool get_hologram_record_requested() const { return hologram_record_requested_; }
    bool get_raw_record_requested() const { return raw_record_requested_; }
    bool get_cuts_record_requested() const { return cuts_record_requested_; }

  public:
    bool update_time_transformation_size(const unsigned short time_transformation_size);

    /*! \name Resources management
     * \{
     */
    void update_spatial_transformation_parameters();
    void init_cuts();
    void dispose_cuts();
    /*! \} */

  protected:
    /*! \brief Reference on the input queue */
    BatchInputQueue& gpu_input_queue_;

    /*! \brief Reference on the output queue */
    Queue& gpu_output_queue_;

    /*! \brief Main buffers. */
    CoreBuffersEnv buffers_;

    /*! \brief Batch environment */
    BatchEnv batch_env_;

    /*! \brief STFT environment. */
    TimeTransformationEnv time_transformation_env_;

    /*! \brief Frame Record environment (Raw + Hologram + Cuts) */
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
    /* FIXME: not used anywhere */
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
    std::atomic<bool> stft_update_roi_requested_{false};
    std::atomic<bool> chart_display_requested_{false};
    std::atomic<bool> disable_chart_display_requested_{false};
    std::atomic<std::optional<unsigned int>> chart_record_requested_{std::nullopt};
    std::atomic<bool> disable_chart_record_requested_{false};
    std::atomic<std::optional<unsigned int>> output_resize_requested_{std::nullopt};
    std::atomic<bool> filter2d_view_requested_{false};
    std::atomic<bool> disable_filter2d_view_requested_{false};
    std::atomic<bool> termination_requested_{false};
    std::atomic<bool> request_update_time_stride_{false};
    std::atomic<bool> request_disable_lens_view_{false};
    std::atomic<bool> hologram_record_requested_{false};
    std::atomic<bool> raw_record_requested_{false};
    std::atomic<bool> cuts_record_requested_{false};

    AdvancedCache::Cache advanced_cache_;
    ComputeCache::Cache compute_cache_;
    ExportCache::Cache export_cache_;
    CompositeCache::Cache composite_cache_;
    Filter2DCache::Cache filter2d_cache_;
    ViewCache::Cache view_cache_;
    ZoneCache::Cache zone_cache_;
    RequestCache::Cache unknown_cache_;
};
} // namespace holovibes
