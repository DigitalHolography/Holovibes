#ifndef PIPELINE_RESSOURCES_HH
# define PIPELINE_RESSOURCES_HH

# include <cufft.h>

# include "queue.hh"

namespace holovibes
{
  class PipelineRessources
  {
  public:
    static const unsigned short sqrt_vector_size = 65535;
  public:
    /*! n stands for the 'nframes' parameters of ComputeDescriptor. */
    PipelineRessources(
      Queue& input,
      Queue& output,
      unsigned short n);
    virtual ~PipelineRessources();

    // \TODO inline getters.
    Queue& get_input_queue();
    Queue& get_output_queue();
    float* get_sqrt_vector() const;
    unsigned short* get_pbuffer();
    cufftHandle get_plan3d();
    cufftHandle get_plan2d();
    cufftComplex* get_lens();
    unsigned short*& get_output_frame_ptr();

    /*! Update pbuffer/plan3d allocation for n output frames. */
    void update_n_parameter(unsigned short n);

  private:
    /*! Alloc and initialize sqrt_vector. */
    void new_gpu_sqrt_vector(unsigned short n);
    void delete_gpu_sqrt_vector();

    /*! Alloc pbuffer for n output frames. */
    void new_gpu_pbuffer(unsigned short n);
    void delete_gpu_pbuffer();

    void new_plan3d(unsigned short n);
    void delete_plan3d();

    void new_plan2d();
    void delete_plan2d();

    void new_gpu_lens();
    void delete_gpu_lens();

    PipelineRessources& operator=(const PipelineRessources&) = delete;
    PipelineRessources(const PipelineRessources&) = delete;
  private:
    Queue& input_;
    Queue& output_;

    /*! Vector filled with sqrtf values. */
    float* gpu_sqrt_vector_;
    /*! Output buffer containing n frames ordered in frequency (p phase). */
    unsigned short* gpu_pbuffer_;
    /*! CUDA FFT Plan 3D. */
    cufftHandle plan3d_;
    /*! CUDA FFT Plan 2D. */
    cufftHandle plan2d_;
    /*! cufftComplex array containing lens. */
    cufftComplex* gpu_lens_;

    /*! Output frame pointer. */
    unsigned short* gpu_output_frame_;
  };
}

#endif /* !PIPELINE_RESSOURCES_HH */