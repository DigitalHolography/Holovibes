/*!
** \brief Contains functions relative to image accumulation.
*/

#pragma once

#include "cuda_tools/unique_ptr.hh"
#include "function_vector.hh"
#include "frame_desc.hh"
#include "queue.hh"
#include "rect.hh"

namespace holovibes
{
class Queue;
class ComputeDescriptor;
struct CoreBuffersEnv;
struct ImageAccEnv;

/*! \brief Contains all functions and structure for computations variables */
namespace compute
{
/*! \class ImageAccumulation
**
** Class that manages the image accumulation
** It manages its own buffer, initialized when needed
** It should be a member of the Pipe class
*/
class ImageAccumulation
{
  public:
    /*!
    ** \brief Constructor.
    */
    ImageAccumulation(FunctionVector& fn_compute_vect,
                      ImageAccEnv& image_acc_env,
                      const CoreBuffersEnv& buffers,
                      const camera::FrameDescriptor& fd,
                      const holovibes::ComputeDescriptor& cd,
                      const cudaStream_t& stream);

    /*!
    ** \brief Enqueue the image accumulation.
    ** Should be called just after gpu_float_buffer is computed
    */
    void insert_image_accumulation();

    /*! \brief Allocate ressources for image accumulation if requested */
    void init();

    /*! \brief Free ressources for image accumulation */
    void dispose();

    /*! \brief Clear image accumulation queue */
    void clear();

  private:
    /*!
    ** \brief Compute average on one view
    */
    void compute_average(std::unique_ptr<Queue>& gpu_accumulation_queue,
                         float* gpu_input_frame,
                         float* gpu_ouput_average_frame,
                         const unsigned int image_acc_level,
                         const size_t frame_res);

    /*!
    ** \brief Insert the average computation of the float frame.
    */
    void insert_compute_average();

    /*!
    ** \brief Insert the copy of the corrected buffer into the float buffer.
    */
    void insert_copy_accumulation_result();

    /*!
    ** \brief Handle the allocation of a accumulation queue and average frame
    */
    void
    allocate_accumulation_queue(std::unique_ptr<Queue>& gpu_accumulation_queue,
                                cuda_tools::UniquePtr<float>& gpu_average_frame,
                                const unsigned int accumulation_level,
                                const camera::FrameDescriptor fd);

  private: /* Attributes */
    /// Image Accumulation environment
    ImageAccEnv& image_acc_env_;

    /// Vector function in which we insert the processing
    FunctionVector& fn_compute_vect_;

    /// Main buffers
    const CoreBuffersEnv& buffers_;

    /// Describes the frame size
    const camera::FrameDescriptor& fd_;
    /// Compute Descriptor
    const ComputeDescriptor& cd_;
    /// Compute stream to perform  pipe computation
    const cudaStream_t& stream_;
};
} // namespace compute
} // namespace holovibes
