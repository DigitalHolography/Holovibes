/*! \file
 *
 * \brief Implementation of FFT1, FFT2 and STFT algorithms.
 */
#pragma once

#include <cufft.h>

#include "frame_desc.hh"
#include "rect.hh"
#include "cuda_tools\unique_ptr.hh"
#include "cuda_tools\array.hh"
#include "cuda_tools\cufft_handle.hh"
#include "function_vector.hh"
#include "global_state_holder.hh"

namespace holovibes
{
class Queue;
class ComputeDescriptor;
struct BatchEnv;
struct TimeTransformationEnv;
struct CoreBuffersEnv;

namespace compute
{
/*! \class FourierTransform
 *
 * \brief #TODO Add a description for this class
 */
class FourierTransform
{
  public:
    /*! \brief Constructor */
    FourierTransform(FunctionVector& fn_compute_vect,
                     const CoreBuffersEnv& buffers,
                     const camera::FrameDescriptor& fd,
                     holovibes::ComputeDescriptor& cd,
                     cuda_tools::CufftHandle& spatial_transformation_plan,
                     TimeTransformationEnv& time_transformation_env,
                     const cudaStream_t& stream,
                     holovibes::ComputeCache::Cache& compute_cache,
                     ViewCache::Cache& view_cache,
                     Filter2DCache::Cache& filter2d_cache_);

    /*! \brief enqueue functions relative to spatial fourier transforms. */
    void insert_fft();

    /*! \brief enqueue functions that store the p frame after the time transformation. */
    void insert_store_p_frame();

    /*! \brief Get Lens Queue used to display the Fresnel lens. */
    std::unique_ptr<Queue>& get_lens_queue();

    /*! \brief enqueue functions relative to temporal fourier transforms. */
    void insert_time_transform();

    /*! \brief Enqueue functions relative to time transformation cuts display when there are activated */
    void insert_time_transformation_cuts_view();

  private:
    /*! \brief Enqueue the call to filter2d cuda function. */
    void insert_filter2d();

    /*! \brief Compute lens and enqueue the call to fft1 cuda function. */
    void insert_fft1();

    /*! \brief Compute lens and enqueue the call to fft2 cuda function. */
    void insert_fft2();

    /*! \brief Enqueue the Fresnel lens into the Lens Queue.
     *
     * It will enqueue the lens, and normalize it, in order to display it correctly later.
     */
    void enqueue_lens();

    /*! \brief Enqueue stft time filtering. */
    void insert_stft();

    /*! \brief Enqueue functions relative to filtering using diagonalization and eigen values.
     *
     * This should eventually replace stft
     */
    void insert_pca();

    void insert_ssa_stft();

    /*! \brief Roi zone of Filter 2D */
    units::RectFd filter2d_zone_;
    units::RectFd filter2d_subzone_;

    /*! \brief Lens used for fresnel transform (During FFT1 and FFT2) */
    cuda_tools::UniquePtr<cufftComplex> gpu_lens_;
    /*! \brief Size of a size of the lens (lens is always a square) */
    uint lens_side_size_ = {0};
    /*! \brief Lens Queue. Used for displaying the lens. */
    std::unique_ptr<Queue> gpu_lens_queue_;

    /*! \brief Size of the buffer needed by cusolver for internal use */
    int cusolver_work_buffer_size_;
    /*! \brief Buffer needed by cusolver for internal use */
    cuda_tools::UniquePtr<cuComplex> cusolver_work_buffer_;

    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;
    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;
    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Compute Descriptor */
    ComputeDescriptor& cd_;
    /*! \brief Pland 2D. Used by FFTs (1, 2, filter2D). */
    cuda_tools::CufftHandle& spatial_transformation_plan_;
    /*! \brief Time transformation environment. */
    TimeTransformationEnv& time_transformation_env_;
    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    ComputeCache::Cache& compute_cache_;
    ViewCache::Cache& view_cache_;
    Filter2DCache::Cache& filter2d_cache_;
};
} // namespace compute
} // namespace holovibes
