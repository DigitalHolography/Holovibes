#ifndef THREAD_COMPUTE_HH_
# define THREAD_COMPUTE_HH_

# include <thread>
# include <cuda.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include "queue.hh"
# include "fft1.cuh"
# include "fft2.cuh"
# include "preprocessing.cuh"

namespace holovibes
{
  class ThreadCompute
  {
  public:
    ThreadCompute(unsigned int p,
      unsigned int images_nb,
      float lambda,
      float dist,
      Queue& q,
      int type);
    ~ThreadCompute();

    Queue& get_queue();
  private:
    void thread_proc();
    void compute_hologram();

  private:
    unsigned int p_;
    unsigned int images_nb_;
    float lambda_;
    float z_;
    Queue& input_q_;

    bool compute_on_;
    std::thread thread_;
    int type_;

    cufftComplex *lens_;
    cufftHandle plan3d_;
    cufftHandle plan2d_;
    float *sqrt_vec_;
    unsigned short *output_buffer_;
    Queue* output_q_;
    
  };
}

#endif /* !THREAD_COMPUTE_HH_ */