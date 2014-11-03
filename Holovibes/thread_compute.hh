#ifndef THREAD_COMPUTE_HH_
# define THREAD_COMPUTE_HH_

# include <thread>
# include <cuda.h>
# include <cuda_runtime.h>
# include <device_launch_parameters.h>
# include "queue.hh"
# include "compute_descriptor.hh"
# include "fft1.cuh"
# include "fft2.cuh"
# include "preprocessing.cuh"

namespace holovibes
{
  class ThreadCompute
  {
  public:
    ThreadCompute(
      const ComputeDescriptor& desc,
      Queue& q);
    ~ThreadCompute();

    Queue& get_queue();
  private:
    void thread_proc();

  private:
    ComputeDescriptor compute_desc_;
    Queue& input_q_;

    bool compute_on_;
    std::thread thread_;

    Queue* output_q_;
  };
}

#endif /* !THREAD_COMPUTE_HH_ */