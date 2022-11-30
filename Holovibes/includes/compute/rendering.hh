/*! \file
 *
 * \brief Implementation of the rendering features.
 */
#pragma once

#include <atomic>

#include "frame_desc.hh"
#include "function_vector.hh"
#include "queue.hh"
#include "rect.hh"
#include "shift_corners.cuh"
#include "global_state_holder.hh"
#include "icompute.hh"

namespace holovibes
{
class ICompute;
struct CoreBuffersEnv;
struct ChartEnv;
struct TimeTransformationEnv;
struct ImageAccEnv;
} // namespace holovibes

namespace holovibes::compute
{
using uint = unsigned int;

/*! \class Rendering
 *
 * \brief #TODO Add a description for this class
 */
class Rendering
{
  public:
    /*! \brief Constructor */
    Rendering(FunctionVector& fn_compute_vect,
              const CoreBuffersEnv& buffers,
              ChartEnv& chart_env,
              const ImageAccEnv& image_acc_env,
              const TimeTransformationEnv& time_transformation_env,
              const FrameDescriptor& input_fd,
              const FrameDescriptor& output_fd,
              const cudaStream_t& stream,
              PipeAdvancedCache& advanced_cache,
              PipeComputeCache& compute_cache,
              PipeExportCache& export_cache,
              PipeViewCache& view_cache,
              PipeZoneCache& zone_cache);
    ~Rendering();

    /*! \brief insert the functions relative to the fft shift. */
    void insert_fft_shift();
    /*! \brief insert the functions relative to noise and signal chart. */
    void insert_chart();
    /*! \brief insert the functions relative to the log10. */
    void insert_log();
    /*! \brief insert the functions relative to the contrast. */
    void insert_contrast();
    void insert_clear_image_accumulation();

  public:
    void request_view_exec_contrast(WindowKind window) { view_exec_contrast_[static_cast<int>(window)] = true; }
    void request_view_clear_image_accumulation(WindowKind window)
    {
        view_clear_image_accumulation_[static_cast<int>(window)] = true;
    }

  protected:
    bool has_requested_view_exec_contrast(WindowKind window) { return view_exec_contrast_[static_cast<int>(window)]; }
    bool has_requested_view_clear_image_accumulation(WindowKind window)
    {
        return view_clear_image_accumulation_[static_cast<int>(window)];
    }

    void reset_view_exec_contrast(WindowKind window) { view_exec_contrast_[static_cast<int>(window)] = false; }
    void reset_view_clear_image_accumulation(WindowKind window)
    {
        view_clear_image_accumulation_[static_cast<int>(window)] = false;
    }

  private:
    std::atomic_bool view_exec_contrast_[4] = {false, false, false, false};
    std::atomic_bool view_clear_image_accumulation_[4] = {false, false, false, false};

  private:
    /*! \brief insert the log10 on the XY window */
    void insert_main_log();
    /*! \brief insert the log10 on the slices */
    void insert_slice_log();
    /*! \brief insert the log10 on the ViewFilter2D view */
    void insert_filter2d_view_log();

    /*! \brief insert the automatic request of contrast */
    void insert_request_exec_contrast();

    /*! \brief insert the constrast on a view */
    void insert_apply_contrast(WindowKind view);

    /*! \brief Calls autocontrast and set the correct contrast variables */
    void autocontrast_caller(float* input, const uint width, const uint height, const uint offset, WindowKind view);

    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;
    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;
    /*! \brief Chart variables */
    ChartEnv& chart_env_;
    /*! \brief Time transformation environment */
    const TimeTransformationEnv& time_transformation_env_;
    /*! \brief Image accumulation environment */
    [[maybe_unused]] const ImageAccEnv& image_acc_env_;
    /*! \brief Describes the input frame size */
    const FrameDescriptor& input_fd_;
    /*! \brief Describes the output frame size */
    const FrameDescriptor& fd_;
    /*! \brief Compute stream to perform  pipe ComputeModeEnum */
    const cudaStream_t& stream_;

    /*! \brief Variables needed for the ComputeModeEnum in the pipe, updated at each end of pipe */
    PipeAdvancedCache& advanced_cache_;
    PipeComputeCache& compute_cache_;
    PipeExportCache& export_cache_;
    PipeViewCache& view_cache_;
    PipeZoneCache& zone_cache_;

    float* percent_min_max_;
};
} // namespace holovibes::compute
