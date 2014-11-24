#ifndef PIPELINE_RESOURCES_HH
# define PIPELINE_RESOURCES_HH

# include <cufft.h>

# include "queue.hh"

namespace holovibes
{
  class PipelineResources
  {
  public:
    static const unsigned short sqrt_vector_size = 65535;
  public:
    /*! n stands for the 'nframes' parameters of ComputeDescriptor. */
    PipelineResources(
      Queue& input,
      Queue& output,
      unsigned short n);
    virtual ~PipelineResources();

    Queue& get_input_queue()
    {
      return input_;
    }
    Queue& get_output_queue()
    {
      return output_;
    }
    float* get_sqrt_vector() const
    {
      return gpu_sqrt_vector_;
    }
    unsigned short* get_output_buffer()
    {
      return gpu_output_buffer_;
    }
    float* get_float_buffer()
    {
      return gpu_float_buffer_;
    }
    cufftHandle get_plan3d()
    {
      return plan3d_;
    }
    cufftHandle get_plan2d()
    {
      return plan2d_;
    }
    cufftComplex* get_lens()
    {
      return gpu_lens_;
    }
    cufftComplex* get_input_buffer()
    {
      return gpu_input_buffer_;
    }
    cufftComplex* get_input_frame_ptr()
    {
      return gpu_input_frame_;
    }
    void set_input_frame_ptr(cufftComplex* ptr)
    {
      gpu_input_frame_ = ptr;
    }
    unsigned short* get_output_frame_ptr()
    {
      return gpu_output_frame_;
    }
    void set_output_frame_ptr(unsigned short* ptr)
    {
      gpu_output_frame_ = ptr;
    }

    /*! Update pbuffer/plan3d allocation for n output frames. */
    void update_n_parameter(unsigned short n);

  private:
    /*! Alloc and initialize sqrt_vector. */
    void new_gpu_sqrt_vector(unsigned short n);
    void delete_gpu_sqrt_vector();

    /*! Alloc pbuffer for n output frames. */
    void new_gpu_output_buffer(unsigned short n);
    void delete_gpu_output_buffer();

    void new_gpu_float_buffer();
    void delete_gpu_float_buffer();

    void new_plan3d(unsigned short n);
    void delete_plan3d();

    void new_plan2d();
    void delete_plan2d();

    void new_gpu_lens();
    void delete_gpu_lens();

    void new_gpu_input_buffer(unsigned short n);
    void delete_gpu_input_buffer();

    PipelineResources& operator=(const PipelineResources&) = delete;
    PipelineResources(const PipelineResources&) = delete;
  private:
    Queue& input_;
    Queue& output_;

    /*! Vector filled with sqrtf values. */
    float* gpu_sqrt_vector_;
    /*! Output frame containing n frames ordered in frequency. */
    unsigned short* gpu_output_buffer_;
    /*! Float frame */
    float* gpu_float_buffer_;
    /*! CUDA FFT Plan 3D. */
    cufftHandle plan3d_;
    /*! CUDA FFT Plan 2D. */
    cufftHandle plan2d_;
    /*! cufftComplex array containing lens. */
    cufftComplex* gpu_lens_;
    /*! cufftComplex array containing n contiguous frames. */
    cufftComplex* gpu_input_buffer_;

    /*! Input frame pointer. */
    cufftComplex* gpu_input_frame_;
    /*! Output frame pointer. */
    unsigned short* gpu_output_frame_;
  };
}

#endif /* !PIPELINE_RESOURCES_HH */