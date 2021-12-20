/*! \file
 *
 * \brief Implementation of postprocessing features on complex buffers.
 */
#pragma once

#include <vector>

#include "function_vector.hh"
#include "frame_desc.hh"
#include "unique_ptr.hh"
#include "cufft_handle.hh"
#include "global_state_holder.hh"

using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
struct CoreBuffersEnv;

namespace compute
{
/*! \class Postprocessing
 *
 * \brief #TODO Add a description for this class
 */
class Postprocessing
{
  public:
    /*! \brief Constructor */
    Postprocessing(FunctionVector& fn_compute_vect,
                   CoreBuffersEnv& buffers,
                   const camera::FrameDescriptor& fd,
                   const cudaStream_t& stream,
                   ComputeCache::Cache& compute_cache,
                   ViewCache::Cache& view_cache,
                   AdvancedCache::Cache& advanced_cache);

    /*! \brief Initialize convolution by allocating the corresponding buffer */
    void init();

    /*! \brief Free the ressources for the postprocessing */
    void dispose();

    /*! \brief Insert the Convolution function. TODO: Check if it works. */
    void insert_convolution();

    /*! \brief Insert the normalization function. */
    void insert_renormalize();

  private:
    /*! \brief Used only when the image is composite convolution to do a convolution on each component */
    void convolution_composite();

    cuda_tools::UniquePtr<cuComplex> gpu_kernel_buffer_;
    cuda_tools::UniquePtr<cuComplex> cuComplex_buffer_;
    cuda_tools::UniquePtr<float> hsv_arr_;

    /*! \brief Result of the reduce operation of the current frame used to renormalize the frames */
    cuda_tools::UniquePtr<double> reduce_result_;

    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;

    /*! \brief Main buffers */
    CoreBuffersEnv& buffers_;

    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;

    /*! \brief Plan used for the convolution (frame width, frame height, cufft_c2c) */
    CufftHandle convolution_plan_;

    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    /*! \brief All view related variables, updated at each end of pipe */
    ComputeCache::Cache& compute_cache_;
    ViewCache::Cache& view_cache_;
    AdvancedCache::Cache& advanced_cache_;
};
} // namespace compute
} // namespace holovibes
