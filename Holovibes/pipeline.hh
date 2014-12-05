#ifndef PIPELINE_HH
# define PIPELINE_HH

# include <vector>
# include <functional>
# include <cufft.h>

# include "queue.hh"
# include "compute_descriptor.hh"

namespace holovibes
{
  class Pipeline
  {
    using FnVector = std::vector<std::function<void()>>;
  public:
    Pipeline(
      Queue& input,
      Queue& output,
      ComputeDescriptor& desc);
    virtual ~Pipeline();

    void request_refresh();
    void request_autofocus();
    void request_autocontrast();
    void request_update_n(unsigned short n);
    void exec();
  private:
    void update_n_parameter(unsigned short n);
    void refresh();

    Pipeline& operator=(const Pipeline&) = delete;
    Pipeline(const Pipeline&) = delete;
  private:
    FnVector fn_vect_;
    ComputeDescriptor& compute_desc_;
    Queue& input_;
    Queue& output_;

    /*! cufftComplex array containing n contiguous frames. */
    cufftComplex* gpu_input_buffer_;
    /*! Output frame containing n frames ordered in frequency. */
    unsigned short* gpu_output_buffer_;
    /*! Float frame */
    float* gpu_float_buffer_;
    /*! Vector filled with sqrtf values. */
    float* gpu_sqrt_vector_;
    /*! cufftComplex array containing lens. */
    cufftComplex* gpu_lens_;
    /*! CUDA FFT Plan 3D. */
    cufftHandle plan3d_;
    /*! CUDA FFT Plan 2D. */
    cufftHandle plan2d_;

    /*! Input frame pointer. */
    cufftComplex* gpu_input_frame_ptr_;

    bool autofocus_requested_;
    bool autocontrast_requested_;
    bool refresh_requested_;
    bool update_n_requested_;

    // FIXME
    std::vector<float> average_results_;
  };
}

#endif /* !PIPELINE_HH */
