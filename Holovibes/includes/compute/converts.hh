/* ________________________________________________________ */
/*                  _                _  _                   */
/*    /\  /\  ___  | |  ___  __   __(_)| |__    ___  ___    */
/*   / /_/ / / _ \ | | / _ \ \ \ / /| || '_ \  / _ \/ __|   */
/*  / __  / | (_) || || (_) | \ V / | || |_) ||  __/\__ \   */
/*  \/ /_/   \___/ |_| \___/   \_/  |_||_.__/  \___||___/   */
/* ________________________________________________________ */

/*! \file

  Implmentation of the conversions between buffers.*/
#pragma once

#include <memory>

#include <cufft.h>

#include "compute_descriptor.hh"
#include "frame_desc.hh"
#include "queue.hh"
#include "cuda_tools\cufft_handle.hh"
#include "function_vector.hh"
#include "enum_img_type.hh"

namespace holovibes
{
class ComputeDescriptor;
struct CoreBuffersEnv;
struct BatchEnv;
struct TimeTransformationEnv;
struct UnwrappingResources;
struct UnwrappingResources_2d;
namespace compute
{
class Converts
{
  public:
    /** \brief Constructor.

    */
    Converts(FunctionVector& fn_compute_vect,
             const CoreBuffersEnv& buffers,
             const BatchEnv& batch_env,
             const TimeTransformationEnv& time_transformation_env,
             cuda_tools::CufftHandle& plan2d,
             ComputeDescriptor& cd,
             const camera::FrameDescriptor& input_fd,
             const camera::FrameDescriptor& output_fd,
             const cudaStream_t& stream);

    /** \brief Insert functions relative to the convertion Complex => Float

    */
    void insert_to_float(bool unwrap_2d_requested);

    /** \brief Insert functions relative to the convertion Float => Unsigned
       Short

    */
    void insert_to_ushort();

    /*! \brief Insert the conversion Uint(8/16/32) => Complex frame by frame */
    void insert_complex_conversion(Queue& input);

  private:
    /** \brief Set pmin_ and pmax_ according to p accumulation. */
    void insert_compute_p_accu();

    /** \brief Insert the convertion Complex => Modulus */
    void insert_to_modulus();

    /** \brief Insert the convertion Complex => Squared Modulus */
    void insert_to_squaredmodulus();

    /** \brief Insert the convertion Complex => Composite */
    void insert_to_composite();

    /** \brief Insert the convertion Complex => Argument */
    void insert_to_argument(bool unwrap_2d_requested);

    /** \brief Insert the convertion Complex => Phase increase */
    void insert_to_phase_increase(bool unwrap_2d_requested);

    /** \brief Insert the convertion Float => Unsigned Short in XY window */
    void insert_main_ushort();

    /** \brief Insert the convertion Float => Unsigned Short in slices. */
    void insert_slice_ushort();

    //! pindex.
    unsigned short pmin_;
    //! maximum value of p accumulation
    unsigned short pmax_;

    /// Vector function in which we insert the processing
    FunctionVector& fn_compute_vect_;

    //! Main buffers
    const CoreBuffersEnv& buffers_;
    //! Batch environment.
    const BatchEnv& batch_env_;
    //! Time transformation environment
    const TimeTransformationEnv& time_transformation_env_;
    //! Phase unwrapping 1D. Used for phase increase and Argument.
    std::unique_ptr<UnwrappingResources> unwrap_res_;
    //! Phase unwrapping 2D. Used for phase increase and Argument.
    std::unique_ptr<UnwrappingResources_2d> unwrap_res_2d_;
    //! Plan 2D. Used for unwrapping.
    cuda_tools::CufftHandle& plan_unwrap_2d_;
    /// Describes the input frame size
    const camera::FrameDescriptor& fd_;
    /// Describes the output frame size
    const camera::FrameDescriptor& output_fd_;
    /// Variables needed for the computation in the pipe
    ComputeDescriptor& cd_;
    /// Compute stream to perform  pipe computation
    const cudaStream_t& stream_;
};
} // namespace compute
} // namespace holovibes
