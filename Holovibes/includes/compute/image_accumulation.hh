/*! \file
 *
 * \brief Contains functions relative to image accumulation.
 */

#pragma once

#include "cuda_tools/unique_ptr.hh"
#include "function_vector.hh"
#include "frame_desc.hh"
#include "queue.hh"
#include "rect.hh"

#include "global_state_holder.hh"

namespace holovibes
{
class Queue;
struct CoreBuffersEnv;
struct ImageAccEnv;
} // namespace holovibes

/*! \brief Contains all functions and structure for computations variables */
namespace holovibes::compute
{
/*! \class ImageAccumulation
 *
 * \brief Class that manages the image accumulation
 *
 * It manages its own buffer, initialized when needed
 * It should be a member of the Pipe class
 */
class ImageAccumulation
{
  public:
    /*! \brief Constructor */
    ImageAccumulation(FunctionVector& fn_compute_vect,
                      ImageAccEnv& image_acc_env,
                      const CoreBuffersEnv& buffers,
                      const camera::FrameDescriptor& fd,
                      const cudaStream_t& stream,
                      ViewCache::Cache& view_cache);

    /*! \brief Enqueue the image accumulation.
     *
     * Should be called just after gpu_float_buffer is computed
     */
    void insert_image_accumulation();

    /*! \brief Allocate ressources for image accumulation if requested */
    void init();

    /*! \brief Free ressources for image accumulation */
    void dispose();

    /*! \brief Clear image accumulation queue */
    void clear();

  private:
    /*! \brief Compute average on one view */
    void compute_average(std::unique_ptr<Queue>& gpu_accumulation_queue,
                         float* gpu_input_frame,
                         float* gpu_ouput_average_frame,
                         const unsigned int image_acc_level,
                         const size_t frame_res);

    /*! \brief Insert the average computation of the float frame. */
    void insert_compute_average();

    /*! \brief Insert the copy of the corrected buffer into the float buffer. */
    void insert_copy_accumulation_result();

    /*! \brief Handle the allocation of a accumulation queue and average frame */
    void allocate_accumulation_queue(std::unique_ptr<Queue>& gpu_accumulation_queue,
                                     cuda_tools::UniquePtr<float>& gpu_average_frame,
                                     const unsigned int accumulation_level,
                                     const camera::FrameDescriptor fd);

  private:
    /*! \brief Vector function in which we insert the processing */
    FunctionVector& fn_compute_vect_;

    /*! \brief Image Accumulation environment */
    ImageAccEnv& image_acc_env_;

    /*! \brief Main buffers */
    const CoreBuffersEnv& buffers_;

    /*! \brief Describes the frame size */
    const camera::FrameDescriptor& fd_;
    /*! \brief Compute stream to perform  pipe computation */
    const cudaStream_t& stream_;

    ViewCache::Cache& view_cache_;
};
} // namespace holovibes::compute
