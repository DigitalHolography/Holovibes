/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file

   Implementation of the rendering features. */
#pragma once

#include <atomic>

#include "frame_desc.hh"
#include "function_vector.hh"
#include "queue.hh"
#include "rect.hh"
#include "shift_corners.cuh"

namespace holovibes
{
class ComputeDescriptor;
class ICompute;
struct CoreBuffersEnv;
struct ChartEnv;
struct TimeTransformationEnv;
struct ImageAccEnv;

namespace compute
{
using uint = unsigned int;

class Rendering
{
  public:
    /** \brief Constructor.

    */
    Rendering(FunctionVector& fn_compute_vect,
              const CoreBuffersEnv& buffers,
              ChartEnv& chart_env,
              const ImageAccEnv& image_acc_env,
              const TimeTransformationEnv& time_transformation_env,
              ComputeDescriptor& cd,
              const camera::FrameDescriptor& input_fd,
              const camera::FrameDescriptor& output_fd,
              ICompute* Ic,
              const cudaStream_t& stream);

    /** \brief insert the functions relative to the fft shift.

    */
    void insert_fft_shift();
    /** \brief insert the functions relative to noise and signal chart.

     */
    void insert_chart();
    /** \brief insert the functions relative to the log10.

    */
    void insert_log();
    /** \brief insert the functions relative to the contrast.

    */
    void insert_contrast(std::atomic<bool>& autocontrast_request,
                         std::atomic<bool>& autocontrast_slice_xz_request,
                         std::atomic<bool>& autocontrast_slice_yz_request);

  private:
    /** \brief insert the log10 on the XY window

    */
    void insert_main_log();
    /** \brief insert the log10 on the slices

    */
    void insert_slice_log();

    /** \brief insert the autocontrast computation

    */
    void insert_compute_autocontrast(
        std::atomic<bool>& autocontrast_request,
        std::atomic<bool>& autocontrast_slice_xz_request,
        std::atomic<bool>& autocontrast_slice_yz_request);

    /** \brief insert the constrast on a view

    */
    void Rendering::insert_apply_contrast(WindowKind view);

    /** \brief Calls autocontrast and set the correct contrast variables

    */
    void autocontrast_caller(float* input,
                             const uint width,
                             const uint height,
                             const uint offset,
                             WindowKind view);

    /** \brief Set the maximum and minimum contrast boundaries (according to the
       percentile)

    */
    void set_contrast_min_max(const float* const percent_out,
                              std::atomic<float>& contrast_min,
                              std::atomic<float>& contrast_max);

    /// Vector function in which we insert the processing
    FunctionVector& fn_compute_vect_;
    /// Main buffers
    const CoreBuffersEnv& buffers_;
    /// Chart variables
    ChartEnv& chart_env_;
    /// Time transformation environment
    const TimeTransformationEnv& time_transformation_env_;
    /// Image accumulation environment
    const ImageAccEnv& image_acc_env_;
    /// Describes the input frame size
    const camera::FrameDescriptor& input_fd_;
    /// Describes the output frame size
    const camera::FrameDescriptor& fd_;
    /// Variables needed for the computation in the pipe
    ComputeDescriptor& cd_;
    /// Pointer on the parent.
    ICompute* Ic_;
    /// Compute stream to perform  pipe computation
    const cudaStream_t& stream_;
};
} // namespace compute
} // namespace holovibes
