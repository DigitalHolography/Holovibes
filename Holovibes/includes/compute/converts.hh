/*! \file
 *
 * \brief Implmentation of the conversions between buffers.
 */
#pragma once

#include <memory>

#include <cufft.h>

#include "frame_desc.hh"
#include "batch_input_queue.hh"
#include "cuda_tools\cufft_handle.hh"
#include "function_vector.hh"
#include "enum_img_type.hh"

namespace holovibes
{
struct CoreBuffersEnv;
struct BatchEnv;
struct TimeTransformationEnv;
struct UnwrappingResources;
struct UnwrappingResources_2d;
} // namespace holovibes

namespace holovibes::compute
{
/*! \class Converts
 *
 * \brief #TODO Add a description for this class
 */
class Converts
{
  public:
    /*! \brief Constructor */
    Converts(FunctionVector& fn_compute_vect,
             const CoreBuffersEnv& buffers,
             const TimeTransformationEnv& time_transformation_env,
             cuda_tools::CufftHandle& plan2d,
             const camera::FrameDescriptor& input_fd,
             const cudaStream_t& stream,
             ViewCache::Cache& view_cache);

    /*! \brief Insert functions relative to the convertion Complex => Float */
    void insert_to_float(bool unwrap_2d_requested,
                         ImgType img_type,
                         TimeTransformation time_transformation,
                         float* buffers_gpu_postprocess_frame,
                         ViewPQ p,
                         uint time_transformation_size,
                         holovibes::CompositeRGB rgb,
                         CompositeKind composite_kind,
                         bool composite_auto_weights,
                         const holovibes::CompositeHSV& composite_hsv,
                         holovibes::units::RectFd composite_zone,
                         unsigned int unwrap_history_size);

    /*! \brief Insert functions relative to the convertion Float => Unsigned Short */
    void insert_to_ushort(bool filter2d_view_enabled);

    /*! \brief Insert the conversion Uint(8/16/32) => Complex frame by frame */
    void insert_complex_conversion(BatchInputQueue& input);

  private:
    /*! \brief Set pmin_ and pmax_ according to p accumulation. */
    void insert_compute_p_accu(uint time_transformation_size, ViewPQ p);

    /*! \brief Insert the convertion Complex => Modulus */
    void insert_to_modulus(float* gpu_postprocess_frame);

    /*! \brief Insert the convertion Complex => Squared Modulus */
    void insert_to_squaredmodulus(float* gpu_postprocess_frame);

    /*! \brief Insert the convertion Complex => Composite */
    void insert_to_composite(holovibes::CompositeRGB rgb,
                             uint time_transformation_size,
                             holovibes::CompositeKind composite_kind,
                             float* gpu_postprocess_frame,
                             bool composite_auto_weights,
                             const holovibes::CompositeHSV& hsv,
                             holovibes::units::RectFd composite_zone);

    /*! \brief Insert the convertion Complex => Argument */
    void insert_to_argument(bool unwrap_2d_requested, float* gpu_postprocess_frame);

    /*! \brief Insert the convertion Complex => Phase increase */
    void insert_to_phase_increase(bool unwrap_2d_requested, unsigned int unwrap_history_size, float* gpu_postprocess_frame);

    /*! \brief Insert the convertion Float => Unsigned Short in XY window */
    void insert_main_ushort();

    /*! \brief Insert the convertion Float => Unsigned Short in slices. */
    void insert_slice_ushort();

    /*! \brief Insert the convertion Float => Unsigned Short of Filter2D View. */
    void insert_filter2d_ushort();

    /*! \brief p_index */
    unsigned short pmin_;
    /*! \brief Maximum value of p accumulation */
    unsigned short pmax_;

    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;

    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;
    /*! \brief Time transformation environment */
    const TimeTransformationEnv& time_transformation_env_;
    /*! \brief Phase unwrapping 1D. Used for phase increase and Argument. */
    std::unique_ptr<UnwrappingResources> unwrap_res_;
    /*! \brief Phase unwrapping 2D. Used for phase increase and Argument. */
    std::unique_ptr<UnwrappingResources_2d> unwrap_res_2d_;
    /*! \brief Plan 2D. Used for unwrapping. */
    cuda_tools::CufftHandle& plan_unwrap_2d_;
    /*! \brief Describes the input frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Compute stream to perform pipe computation */
    const cudaStream_t& stream_;

    /*! \brief Variables needed for the computation in the pipe, updated at each end of pipe */
    ViewCache::Cache& view_cache_;
};
} // namespace holovibes::compute
