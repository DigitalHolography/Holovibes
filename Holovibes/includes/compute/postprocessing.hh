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
using holovibes::cuda_tools::CufftHandle;

namespace holovibes
{
class ComputeDescriptor;
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
    /** \brief Constructor. */
    Postprocessing(FunctionVector& fn_compute_vect,
                   CoreBuffersEnv& buffers,
                   const camera::FrameDescriptor& fd,
                   holovibes::ComputeDescriptor& cd,
                   const cudaStream_t& stream);

    /*! \brief Initialize convolution by allocating the corresponding
    ** buffer
    */
    void init();

    /*! \brief Free the ressources for the postprocessing */
    void dispose();

    /*! \brief Insert the Convolution function. TODO: Check if it works. */
    void insert_convolution();

    /*! \brief Insert the normalization function. */
    void insert_renormalize();

  private:
    //! used only when the image is composite convolution to do a convolution on
    //! each component
    void convolution_composite();

    cuda_tools::UniquePtr<cuComplex> gpu_kernel_buffer_;
    cuda_tools::UniquePtr<cuComplex> cuComplex_buffer_;
    cuda_tools::UniquePtr<float> hsv_arr_;

    //! Result of the reduce operation of the current frame used to renormalize
    //! the frames
    cuda_tools::UniquePtr<double> reduce_result_;

    //! Vector function in which we insert the processing
    FunctionVector& fn_compute_vect_;

    //! Main buffers
    CoreBuffersEnv& buffers_;

    // Describes the frame size
    const camera::FrameDescriptor& fd_;

    //! Compute Descriptor
    ComputeDescriptor& cd_;

    // plan used for the convolution (frame width, frame height, cufft_c2c)
    CufftHandle convolution_plan_;

    /// Compute stream to perform  pipe computation
    const cudaStream_t& stream_;
};
} // namespace compute
} // namespace holovibes
