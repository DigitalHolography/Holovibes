/*! \file
 *
 * Contains functions relative to image XY stabilization. */
#pragma once

#include "cuda_tools/unique_ptr.hh"
#include "cuda_tools/array.hh"
#include "cuda_tools/nppi_data.hh"
#include "aliases.hh"
#include "frame_desc.hh"
#include "queue.hh"
#include "rect.hh"

namespace holovibes
{
class ComputeDescriptor;
struct CoreBuffers;
} // namespace holovibes

/*! \brief Contains all functions and structure for computations variables */
namespace holovibes::compute
{

/*! \class Stabilization
**
** Class that manages the stabilization of the image
** It manages its own buffer, initialized when needed
** It should be a member of the Pipe class
*/
class Stabilization
{
  public:
    /** \brief Constructor.

    */
    Stabilization(FnVector& fn_vect,
                  const CoreBuffers& buffers,
                  const camera::FrameDescriptor& fd,
                  const holovibes::ComputeDescriptor& cd);

    /*! \brief Enqueue the functions relative to XY correction.
    **
    ** Should be called just after gpu_float_buffer is computed
    */
    void insert_post_img_type();

  private:
    /*! \brief Insert the computation of the average of the float frame.

    */
    void insert_average_compute();
    /*! \brief Insert the correlation function.

    */
    void insert_correlation();
    /*! \brief Insert the computation of the extrema.

    */
    void insert_extremums();
    /*! \brief Insert the main stabilization function.

    */
    void insert_stabilization();
    /*! \brief Insert the copy of the corrected buffer into the float buffer.

    */
    void insert_float_buffer_overwrite();

    /** \brief Computes the correlation between two buffers and write the result into convolution_.

    */
    void compute_correlation(const float* x, const float* y);
    /** \brief Computes the convolution between two buffers and write the result into \param out

    */
    void compute_convolution(const float* x, const float* y, float* out);

    /// Buffer to keep the convolution product
    cuda_tools::Array<float> convolution_;

    /// Buffer used to temporaly store the average, to compare it with current frame
    cuda_tools::UniquePtr<float> float_buffer_average_;

    //! Shift computed to correct the error on x-axis.
    int shift_x;
    //! Shift computed to correct the error on y-axis.
    int shift_y;

    //! Queue accumulating XY frames.
    std::unique_ptr<Queue> accumulation_queue_;

    /// Vector function in which we insert the processing
    FnVector& fn_vect_;
    //! Main buffers
    const CoreBuffers& buffers_;
    /// Describes the frame size
    const camera::FrameDescriptor& fd_;

    //! Compute Descriptor
    const ComputeDescriptor& cd_;

    //! Nppi data used for Nppi function calls
    cuda_tools::NppiData nppi_data_;
};
} // namespace holovibes::compute