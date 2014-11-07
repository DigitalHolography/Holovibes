#include "stdafx.h"
#include "thread_compute.hh"

namespace holovibes
{
  ThreadCompute::ThreadCompute(
    const ComputeDescriptor& desc,
    Queue& input_q)
    : compute_desc_(desc)
    , input_q_(input_q)
    , compute_on_(true)
    , thread_(&ThreadCompute::thread_proc, this)
  {
    camera::FrameDescriptor fd = input_q_.get_frame_desc();
    fd.depth = 2;

    output_q_ = new Queue(fd, input_q_.get_max_elts());
  }

  ThreadCompute::~ThreadCompute()
  {
    compute_on_ = false;

    if (thread_.joinable())
      thread_.join();

    delete output_q_;
  }

  Queue& ThreadCompute::get_queue()
  {
    return *output_q_;
  }

  void ThreadCompute::thread_proc()
  {
    /* Ressources allocation */
    float* sqrt_array = make_sqrt_vec(65536);
    /* Output buffer containing p images ordered in frequency. */
    unsigned short *pbuffer;
    cudaMalloc(
      &pbuffer,
      input_q_.get_pixels() * sizeof(unsigned short) * compute_desc_.nsamples);
    cufftHandle plan;
    cufftPlan3d(
      &plan,
      compute_desc_.nsamples,            // NX
      input_q_.get_frame_desc().width,   // NY
      input_q_.get_frame_desc().height,  // NZ
      CUFFT_C2C);

    cufftComplex* lens = nullptr;

    /* Pointer on the selected FFT algorithm */
    void(*fft_algorithm)(
      int nsamples,
      holovibes::Queue* q,
      cufftComplex* lens,
      float* sqrt_array,
      unsigned short* pbuffer,
      cufftHandle plan) = nullptr;

    if (compute_desc_.algorithm == ComputeDescriptor::FFT1)
    {
      lens = create_lens(
        input_q_.get_frame_desc(),
        compute_desc_.lambda,
        compute_desc_.zdistance);
      fft_algorithm = &fft_1;
    }
    else if (compute_desc_.algorithm == ComputeDescriptor::FFT2)
    {
      lens = create_spectral(
        compute_desc_.lambda,
        compute_desc_.zdistance,
        input_q_.get_frame_desc().width,
        input_q_.get_frame_desc().height,
        input_q_.get_frame_desc().pixel_size,
        input_q_.get_frame_desc().pixel_size);
      fft_algorithm = &fft_2;
    }
    else
      assert(!"Impossible case");

    /* Thread loop */
    while (compute_on_)
    {
      if (input_q_.get_current_elts() >= compute_desc_.nsamples)
      {
        fft_algorithm(
          compute_desc_.nsamples,
          &input_q_,
          lens,
          sqrt_array,
          pbuffer,
          plan);

        /* Shifting */
        unsigned short *shifted = pbuffer + compute_desc_.pindex * input_q_.get_pixels();
        shift_corners(&shifted, output_q_->get_frame_desc().width, output_q_->get_frame_desc().height);

        /* Store p-th image */
        output_q_->enqueue(shifted, cudaMemcpyDeviceToDevice);
        input_q_.dequeue();
      }
    }

    /* Free ressources */
    cudaFree(lens);
    cufftDestroy(plan);
    cudaFree(pbuffer);
    cudaFree(sqrt_array);
  }
}