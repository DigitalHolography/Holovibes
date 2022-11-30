/*! \file
 *
 * \brief Stores functions helping the editing of the images.
 */
#pragma once

#include "env_structs.hh"
#include "all_pipe_requests_on_sync_functions.hh"
#include "pipe_request_on_sync.hh"

namespace holovibes
{
// clang-format off
using PipeAdvancedCache = AdvancedCache::Cache<PipeRequestOnSyncWrapper<AdvancedPipeRequestOnSync, AdvancedCacheFrontEndMethods>>;
using PipeComputeCache = ComputeCache::Cache<PipeRequestOnSyncWrapper<ComputePipeRequestOnSync, ComputeCacheFrontEndMethods>>;
using PipeImportCache = ImportCache::Cache<PipeRequestOnSyncWrapper<ImportPipeRequestOnSync, ImportCacheFrontEndMethods>>;
using PipeExportCache = ExportCache::Cache<PipeRequestOnSyncWrapper<ExportPipeRequestOnSync, ExportCacheFrontEndMethods>>;
using PipeCompositeCache = CompositeCache::Cache<PipeRequestOnSyncWrapper<CompositePipeRequestOnSync, CompositeCacheFrontEndMethods>>;
using PipeViewCache = ViewCache::Cache<PipeRequestOnSyncWrapper<ViewPipeRequestOnSync, ViewCacheFrontEndMethods>>;
using PipeZoneCache = ZoneCache::Cache<PipeRequestOnSyncWrapper<DefaultPipeRequestOnSync, ZoneCacheFrontEndMethods>>;
// clang-format on

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
    PipeAdvancedCache& get_advanced_cache() { return advanced_cache_; }
    PipeComputeCache& get_compute_cache() { return compute_cache_; }
    PipeImportCache& get_import_cache() { return import_cache_; }
    PipeExportCache& get_export_cache() { return export_cache_; }
    PipeCompositeCache& get_composite_cache() { return composite_cache_; }
    PipeViewCache& get_view_cache() { return view_cache_; }
    PipeZoneCache& get_zone_cache() { return zone_cache_; }

    BatchInputQueue& get_gpu_input_queue() { return gpu_input_queue_; };
    Queue& get_gpu_output_queue() { return gpu_output_queue_; }
    CoreBuffersEnv& get_buffers() { return buffers_; }
    BatchEnv& get_batch_env() { return batch_env_; }
    TimeTransformationEnv& get_time_transformation_env() { return time_transformation_env_; }
    FrameRecordEnv& get_frame_record_env() { return frame_record_env_; }
    ChartEnv& get_chart_env() { return chart_env_; }
    ImageAccEnv& get_image_acc_env() { return image_acc_env_; }

    std::unique_ptr<Queue>& get_raw_view_queue_ptr() { return gpu_raw_view_queue_; }
    std::unique_ptr<Queue>& get_filter2d_view_queue_ptr() { return gpu_filter2d_view_queue_; }

    void request_termination() { termination_requested_ = true; }

  public:
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

    std::unique_ptr<Queue>& get_stft_slice_queue(int i);

  public:
    bool update_time_transformation_size(uint time_transformation_size);

  private:
    void update_time_transformation_size_resize(uint time_transformation_size);

  public:
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

    /*! \brief Compute stream to perform pipe ComputeModeEnum */
    const cudaStream_t& stream_;

    /*! \brief Chrono counting time between two iteration
     *
     * Taking into account steps, since it is executing at the end of pipe.
     */
    /* FIXME: not used anywhere */
    std::chrono::time_point<std::chrono::steady_clock> past_time_;

    /*! \brief Counting pipe iteration, in order to update fps only every 100 iterations. */
    unsigned int frame_count_{0};

    std::atomic<bool> termination_requested_{false};

    PipeAdvancedCache advanced_cache_;
    PipeComputeCache compute_cache_;
    PipeImportCache import_cache_;
    PipeExportCache export_cache_;
    PipeCompositeCache composite_cache_;
    PipeViewCache view_cache_;
    PipeZoneCache zone_cache_;
};
} // namespace holovibes
